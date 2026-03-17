# Drug Side Effect XGBoost Comparison

본 저장소에서는 `THROMBOCYTOPENIA` 예측을 대상으로 XGBoost 기반 비교 실험을 수행하였다. 구조 정보만 사용하는 설정과 구조 + target action 정보를 함께 사용하는 설정을 동일한 문제에서 비교하여 특징 결합의 효과를 확인하고자 하였다.

## English Summary

This repository contains XGBoost comparison experiments for `THROMBOCYTOPENIA`. It compares structure-only settings against fused structure and target action settings under the same prediction task.

## 연구 개요

| 항목 | 내용 |
| --- | --- |
| 연구 초점 | XGBoost 기반 특징 설정 비교 |
| 입력 데이터 | `../drug-side-effect-shared-datasets/`의 구조 및 통합 데이터셋 |
| 수행 내용 | SMOTE 적용, 특징 조합별 XGBoost 비교 |
| 저장소 역할 | 핵심 모델링 이후의 성능 비교 단계 |

## 주요 결과

| Setting | Accuracy | F1-score | ROC-AUC | Recall |
| --- | ---: | ---: | ---: | ---: |
| `SMILES only (32)` | 0.533 | 0.481 | 0.529 | 0.481 |
| `SMILES only (64)` | 0.550 | 0.491 | 0.544 | 0.481 |
| `SMILES + target (128)` | 0.462 | 0.471 | 0.521 | 0.704 |

보존된 결과에서는 `SMILES only (64)` 설정이 가장 안정적인 F1-score를 보였고, `SMILES + target` 설정은 recall이 증가하여 양성 사례를 더 넓게 포착하는 경향을 보였다.

## 주요 파일

| 파일 | 설명 |
| --- | --- |
| `xgb_maincode.py` | 메인 비교 실험 |
| `xgb_onlysmiles.py` | 구조 정보만 사용한 실험 |
| `xgb_target_smiles.py` | 구조 + target 정보를 결합한 실험 |
