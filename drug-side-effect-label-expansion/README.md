# Drug Side Effect Label Expansion

본 저장소는 보다 큰 ADR 라벨 공간으로 연구를 확장하려는 후기 시도를 정리한 것이다. 초기의 소규모 라벨 예측에서 벗어나, 대형 multi-label ADR 행렬을 다루려는 실험 흔적이 포함되어 있다.

## English Summary

This repository captures a later attempt to extend the study toward a much larger ADR label space. It reflects the transition from a small curated label setting to a broader multi-label setting.

## 연구 개요

| 항목 | 내용 |
| --- | --- |
| 연구 초점 | 후기 ADR label expansion 시도 |
| 입력 데이터 | `label.csv`, `additional_adrs_with_fp.csv`, `target_action_data222.csv` |
| 수행 내용 | 구조 및 target 기반 특징을 대형 label matrix에 적용하려는 탐색 |
| 저장소 역할 | 연구 확장 방향을 보여주는 기록 |

## 주요 파일

| 파일 | 설명 |
| --- | --- |
| `smile.py` | 메인 실험 스크립트 |
| `label.csv` | 대형 ADR label matrix |
| `slimes_data_v2.csv` | 구조 입력 테이블 |
