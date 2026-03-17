# Drug Side Effect Core Modeling

본 저장소에서는 `SMILES`, target action 정보, 승인 시점 메타데이터를 통합하여 학습용 데이터셋을 구성하고, `THROMBOCYTOPENIA` 예측을 중심으로 핵심 모델링 실험을 수행하였다. 전체 연구 흐름에서 가장 중심이 되는 메인 모델링 저장소이다.

## English Summary

This repository is the core modeling component of the study. It integrates `SMILES`, target action features, and approval-time metadata and evaluates the main experiments focused on `THROMBOCYTOPENIA`.

## 연구 개요

| 항목 | 내용 |
| --- | --- |
| 연구 초점 | `THROMBOCYTOPENIA` 중심 통합 모델링 |
| 입력 데이터 | `dataset_0629.csv`, `dataset_left_0629.csv`, `dataset(2)_0629.csv` |
| 수행 내용 | 데이터 병합, 승인일 정렬, SMOTE 적용, Random Forest 및 LightGBM 실험 |
| 저장소 역할 | 전체 연구의 핵심 모델링 단계 |

## 주요 결과

| Result File | Accuracy | F1-score | Recall |
| --- | ---: | ---: | ---: |
| `test_ranfo_0630_result_4fold_smote_testsize3_THROMBOCYTOPENIA.csv` | 0.531 | 0.478 | 0.533 |

핵심 Random Forest 결과는 구조 정보와 target 관련 정보를 함께 사용한 기준선으로 정리되었으며, 이후 XGBoost 비교 실험의 기준점 역할을 하였다.

## 주요 파일

| 파일 | 설명 |
| --- | --- |
| `data_preprocessing.py` | 통합 데이터셋 생성 |
| `approval.py` | 승인일 수집 |
| `download.py` | target 정보 보완용 스크립트 |
| `lgbm_test.py` | LightGBM 실험 |
| `ranfo_0610.py` | Random Forest 실험 |
