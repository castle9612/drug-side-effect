# Drug Side Effect Random Forest Baseline

본 저장소는 초기 단계에서 수행한 Random Forest 기준선 실험을 정리한 것이다. 약물의 구조 정보와 target-action 정보를 결합하여 선별된 ADR 라벨을 예측할 수 있는지 확인하고자 하였다.

## English Summary

This repository contains the early Random Forest baseline experiments. The main goal was to test whether curated ADR labels could be predicted from molecular structure and target-action features.

## 연구 개요

| 항목 | 내용 |
| --- | --- |
| 연구 초점 | 초기 Random Forest 기준선 |
| 입력 데이터 | `adr_selected.csv`, `smile_data_v2.csv`, `smile_data_v3.csv`, `target_action_data_v2.csv` |
| 수행 내용 | Morgan fingerprint, similarity matrix, AutoEncoder compression, Random Forest |
| 저장소 역할 | 후속 모델링 실험의 출발점 |

## 주요 파일

| 파일 | 설명 |
| --- | --- |
| `rf_0417.py` | 메인 기준선 실험 스크립트 |
| `READ_ME.txt` | 실험 메모 |
| `rf_0417_TACHYCADIA.csv` | 예시 결과 파일 |
