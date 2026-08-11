"""실시간 속도 플롯 — 순수 로직(ROS/matplotlib 무관, 단위테스트 대상).

/mpc_debug (Float32MultiArray) 의 v_cmd/v_actual/ref_v 추출 + 롤링 시간창 트림.
"""
from __future__ import annotations


def extract_speeds(data, i_cmd=0, i_actual=2, i_ref=8):
    """Float32MultiArray.data 에서 (v_cmd, v_actual, ref_v) 추출.
    길이가 max(인덱스)+1 미만이면 None (구버전/짧은 배열 가드).
    """
    if len(data) <= max(i_cmd, i_actual, i_ref):
        return None
    return (float(data[i_cmd]), float(data[i_actual]), float(data[i_ref]))


def trim_window(times, t_now, window_sec):
    """오름차순 times 에서 (t_now - window_sec) 보다 오래된 앞쪽 개수 반환
    (노드가 del buf[:k] 로 롤링). 빈 입력/전부 최신 → 0.
    """
    cutoff = float(t_now) - float(window_sec)
    k = 0
    for t in times:
        if float(t) < cutoff:
            k += 1
        else:
            break
    return k
