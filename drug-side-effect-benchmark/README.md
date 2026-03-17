# Drug Side Effect Benchmark Study

본 저장소에서는 사전 생성된 feature table을 이용하여 `THROMBOCYTOPENIA` 예측 성능을 비교하였다. 원시 데이터 전처리보다는, 이미 정리된 특징 공간에서 여러 모델을 비교하는 독립 벤치마크 실험에 가깝다.

## English Summary

This repository evaluates `THROMBOCYTOPENIA` prediction on a precomputed feature table. It is closer to a standalone benchmark than to a raw preprocessing pipeline.

## 연구 개요

| 항목 | 내용 |
| --- | --- |
| 연구 초점 | 사전 계산된 특징 기반 모델 비교 |
| 입력 데이터 | `final_df_sorted_factor_lda.csv` |
| 수행 내용 | AutoEncoder 압축 후 다양한 분류기 비교 |
| 저장소 역할 | 독립 벤치마크 및 보조 비교 실험 |

## 주요 파일

| 파일 | 설명 |
| --- | --- |
| `train_code.py` | 메인 벤치마크 및 하이퍼파라미터 탐색 코드 |
| `adr_test_model.py` | 모델 정의 보조 코드 |
| `adr_test.py` | 추가 실험 스크립트 |
