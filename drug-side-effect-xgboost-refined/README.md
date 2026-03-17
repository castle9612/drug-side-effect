# Drug Side Effect XGBoost Refined Study

본 저장소에서는 `dataset_0722`를 기반으로 후속 XGBoost 실험을 수행하였다. 이전 비교 실험보다 정리된 데이터셋에서 구조 정보와 target 관련 특징의 조합을 다시 평가하고, 실험 조건을 보다 일관되게 정리하고자 하였다.

## English Summary

This repository documents the follow-up XGBoost experiments built on `dataset_0722`. It revisits the combination of structure and target-related features in a more refined dataset setting.

## 연구 개요

| 항목 | 내용 |
| --- | --- |
| 연구 초점 | `dataset_0722` 기반 후속 XGBoost 실험 |
| 입력 데이터 | `../drug-side-effect-shared-datasets/dataset_0722.csv` |
| 수행 내용 | refined dataset 기반 XGBoost 재평가 |
| 저장소 역할 | 후기 비교 실험 정리 |

## 주요 결과

| Result File | Accuracy | F1-score | ROC-AUC | Recall |
| --- | ---: | ---: | ---: | ---: |
| `test_xgboost_0722_result_5fold_smote_smiles_target_128THROMBOCYTOPENIA.csv` | 0.550 | 0.471 | 0.540 | 0.444 |

정제된 데이터셋을 사용한 후속 실험에서는 accuracy는 유지되었으나, 구조 단독 설정 대비 F1-score의 추가 향상은 제한적이었다.

## 주요 파일

| 파일 | 설명 |
| --- | --- |
| `xgboost_0722.py` | 메인 실험 스크립트 |
| `crolling_target.ipynb` | target 수집 보조 노트북 |
| `xgboost_test.ipynb` | 점검용 노트북 |
| `requirements.txt` | 당시 환경 기록 |
