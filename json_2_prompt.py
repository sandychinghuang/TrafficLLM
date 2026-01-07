#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
將 tshark -T json 產物轉成 TrafficLLM 的 <packet> prompts
- 模式:
  * flow   : 以 tcp.stream/udp.stream 聚合，無則退回 (proto, src, dst)
  * packet : 每封包一筆
  * window : 以時間視窗切片，可選 window 分組鍵 (proto / 3tuple)
- 特徵: ip.proto, ip.len, tcp.len, tcp.flags, tcp.window_size / udp.length,
       tls.record.content_type, tls.record.length
- 多層協定: 自動取最後一層 ip/tcp/udp/tls（支援 GRE/隧道）
- 輸出: JSONL，每行 {"flow_id","n_packets","text"}，text 為多行 <packet>: ...
"""
import argparse, json
from collections import defaultdict
from typing import Any, Dict, List, Tuple

def get_layers(obj: Dict[str, Any]) -> Dict[str, Any]:
    if "_source" in obj and "layers" in obj["_source"]:
        return obj["_source"]["layers"]
    return obj.get("layers", {})

def f2float(x, default=0.0):
    try:
        return float(str(x))
    except Exception:
        return default

def collect_layer_variants(layers: Dict[str, Any], base: str) -> List[Dict[str, Any]]:
    cand = []
    for k, v in layers.items():
        if k == base or k.startswith(base + "_"):
            if isinstance(v, dict):
                cand.append(v)
    return cand

def last_layer(layers: Dict[str, Any], base: str) -> Dict[str, Any]:
    c = collect_layer_variants(layers, base)
    return c[-1] if c else (layers.get(base, {}) if isinstance(layers.get(base, {}), dict) else {})

def first_nonempty(*vals):
    for v in vals:
        if v:
            return v
    return None

def collect_tls_records(tls_layer: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    ct_list, ln_list = [], []
    if not isinstance(tls_layer, dict):
        return ct_list, ln_list

    rec = tls_layer.get("tls.record")
    if rec is not None:
        recs = rec if isinstance(rec, list) else [rec]
        for r in recs:
            if isinstance(r, dict):
                ct = r.get("tls.record.content_type")
                ln = r.get("tls.record.length")
                if ct is not None: ct_list.append(str(ct))
                if ln is not None: ln_list.append(str(ln))

    # 兜底：其他變體鍵名
    for k, v in tls_layer.items():
        if "tls.record" in k and k != "tls.record":
            items = v if isinstance(v, list) else [v]
            for r in items:
                if isinstance(r, dict):
                    ct = r.get("tls.record.content_type")
                    ln = r.get("tls.record.length")
                    if ct is not None: ct_list.append(str(ct))
                    if ln is not None: ln_list.append(str(ln))
    return ct_list, ln_list

def packet_line(layers: Dict[str, Any]) -> str:
    ip  = last_layer(layers, "ip")
    tcp = last_layer(layers, "tcp")
    udp = last_layer(layers, "udp")
    tls = last_layer(layers, "tls")
    gre = last_layer(layers, "gre")  # 有些 GRE 會帶資訊，如 gre.proto

    parts: List[str] = []
    # 先放最內層 IP 的 proto/len
    ip_proto = ip.get("ip.proto")
    ip_len   = ip.get("ip.len")
    if ip_proto is not None: parts.append(f"ip.proto: {ip_proto}")
    if ip_len   is not None: parts.append(f"ip.len: {ip_len}")

    # 補充 GRE（可有可無，保留 proto 型別當特徵）
    gre_proto = gre.get("gre.proto")
    if gre_proto is not None:
        parts.append(f"gre.proto: {gre_proto}")

    # TCP / UDP
    if tcp:
        tcp_len = tcp.get("tcp.len")
        if tcp_len is not None: parts.append(f"tcp.len: {tcp_len}")
        tcp_flags = tcp.get("tcp.flags")
        if tcp_flags is not None: parts.append(f"tcp.flags: {tcp_flags}")
        win = first_nonempty(tcp.get("tcp.window_size_value"), tcp.get("tcp.window_size"))
        if win is not None: parts.append(f"tcp.window_size: {win}")
    elif udp:
        ulen = first_nonempty(udp.get("udp.length"), udp.get("udp.len"))
        if ulen is not None: parts.append(f"udp.length: {ulen}")

    # TLS 多 record
    if tls:
        ct_list, ln_list = collect_tls_records(tls)
        if ct_list:
            parts.append("tls.record.content_type: [" + ",".join(ct_list) + "]")
        if ln_list:
            parts.append("tls.record.length: [" + ",".join(ln_list) + "]")

    return "<packet>: " + ", ".join(parts) if parts else ""

# ---------- 分組鍵（只用於聚合，不會輸出到 prompt） ----------
def flow_key_by_stream(layers: Dict[str, Any]) -> str:
    tcp = last_layer(layers, "tcp")
    udp = last_layer(layers, "udp")
    if tcp and "tcp.stream" in tcp:
        return f"tcp:{tcp.get('tcp.stream')}"
    if udp and "udp.stream" in udp:
        return f"udp:{udp.get('udp.stream')}"
    # 退回最內層 IP 的 (proto, src, dst)
    ip = last_layer(layers, "ip")
    proto = str(ip.get("ip.proto", ""))
    src = str(ip.get("ip.src", "") or ip.get("ip.src_host", ""))
    dst = str(ip.get("ip.dst", "") or ip.get("ip.dst_host", ""))
    return f"p3:{proto}:{src}:{dst}"

def group_key_proto(layers: Dict[str, Any]) -> str:
    ip = last_layer(layers, "ip")
    return f"proto:{ip.get('ip.proto','')}"  # 例如 proto:47

# -------------------------------------------------------------
def convert(json_in: str, out_jsonl: str, mode: str = "flow",
            max_packets_per_record: int = 80,
            window_sec: float = 1.0,
            window_key: str = "proto") -> int:

    with open(json_in, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("輸入不是 JSON 陣列（請確認使用 tshark -T json）")

    # 先把每包抽成 (time, layers, frame_no, line)
    packets = []
    for obj in data:
        layers = get_layers(obj)
        if not layers:
            continue
        t = f2float(layers.get("frame", {}).get("frame.time_epoch", "0"))
        frame_no = layers.get("frame", {}).get("frame.number") or ""
        line = packet_line(layers)
        if not line:
            continue
        packets.append((t, layers, str(frame_no), line))

    packets.sort(key=lambda x: x[0])

    n_out = 0
    with open(out_jsonl, "w", encoding="utf-8") as fout:
        if mode == "packet":
            # 每包一筆
            for t, layers, fno, line in packets:
                rec = {"flow_id": f"pkt:{fno or n_out}", "n_packets": 1, "text": line}
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_out += 1

        elif mode == "flow":
            # 依 stream / (proto,src,dst) 聚合
            groups = defaultdict(list)
            for item in packets:
                _, layers, _, _ = item
                gid = flow_key_by_stream(layers)
                groups[gid].append(item)

            for gid, items in groups.items():
                # 取最多 max_packets_per_record 行
                lines = [it[3] for it in items[:max_packets_per_record]]
                if not lines: continue
                rec = {"flow_id": gid, "n_packets": len(lines), "text": "\n".join(lines)}
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_out += 1

        elif mode == "window":
            # 依 key 分組，再用時間視窗切片
            if window_key not in {"proto", "3tuple"}:
                raise ValueError("--window-key 只能是 proto 或 3tuple")
            key_fn = group_key_proto if window_key == "proto" else flow_key_by_stream

            buckets = defaultdict(list)
            for item in packets:
                _, layers, _, _ = item
                buckets[key_fn(layers)].append(item)

            for gid, items in buckets.items():
                start = None
                buf: List[str] = []
                count = 0
                for t, _, _, line in items:
                    if start is None:
                        start = t
                    # 滿視窗或達到最大行數就出一筆
                    if (t - start) > window_sec or count >= max_packets_per_record:
                        if buf:
                            rec = {"flow_id": f"{gid}@{start:.6f}", "n_packets": len(buf), "text": "\n".join(buf)}
                            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                            n_out += 1
                        start = t
                        buf = []
                        count = 0
                    buf.append(line)
                    count += 1
                # 收尾
                if buf:
                    rec = {"flow_id": f"{gid}@{start:.6f}", "n_packets": len(buf), "text": "\n".join(buf)}
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    n_out += 1
        else:
            raise ValueError("未知的 --mode，請用 flow / packet / window")

    return n_out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True, help="tshark -T json 產生的檔案（例如 test.json）")
    ap.add_argument("--output", default="prompts.jsonl", help="輸出 JSONL（每行一筆樣本）")
    ap.add_argument("--mode", default="packet", choices=["flow","packet","window"], help="聚合模式")
    ap.add_argument("--max-per-record", type=int, default=80, help="每筆樣本最多保留的封包行數")
    ap.add_argument("--window-sec", type=float, default=1.0, help="window 模式的時間視窗秒數")
    ap.add_argument("--window-key", default="proto", choices=["proto","3tuple"],
                    help="window 模式切片時的分組鍵：僅以協定(proto)或以 3tuple/stream")
    args = ap.parse_args()

    n = convert(args.json, args.output, mode=args.mode,
                max_packets_per_record=args.max_per_record,
                window_sec=args.window_sec, window_key=args.window_key)
    print(f"✅ 轉換完成：{n} 筆樣本 → {args.output}")

if __name__ == "__main__":
    main()
