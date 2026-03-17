# Drug Side Effect Multilabel Expansion

본 저장소는 연구가 대규모 multi-label ADR 예측으로 확장되는 과정에서 생성된 데이터와 프로토타입 실험을 정리한 것이다. 약물 x ADR 행렬을 구축하고, 이를 바탕으로 후속 모델링 가능성을 탐색하였다.

## English Summary

This repository documents the transition toward large-scale multi-label ADR prediction. It includes the construction of drug-by-ADR matrices and several prototype experiments built on top of them.

## 연구 개요

| 항목 | 내용 |
| --- | --- |
| 연구 초점 | 대규모 ADR matrix 생성 및 후기 확장 실험 |
| 입력 데이터 | `outputdata.csv`, `label.csv`, `drug_adr_sider_freq_v1.csv`, `aftet_siderFreq_v2.csv` |
| 수행 내용 | drug x ADR pivot, label alignment, two-tower style prototype |
| 저장소 역할 | 단일 타깃 예측에서 multi-label 확장으로 넘어가는 단계 |

## 주요 파일

| 파일 | 설명 |
| --- | --- |
| `make_outputdata.py` | ADR matrix 생성 |
| `match.py` | label matrix와 보조 특징 정렬 |
| `twotower.py` | 후속 모델링 아이디어 실험 |
