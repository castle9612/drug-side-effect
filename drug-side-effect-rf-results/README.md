# Drug Side Effect Random Forest Results

본 저장소는 초기 Random Forest 실험에서 생성된 결과 CSV를 정리한 아카이브이다. fold 수와 sampling 방식에 따라 기준선 성능이 어떻게 달라지는지를 비교할 수 있도록 보존하였다.

## English Summary

This repository archives the CSV outputs from the early Random Forest experiments. It records how the baseline changed under different fold settings and sampling strategies.

## 연구 개요

| 항목 | 내용 |
| --- | --- |
| 연구 초점 | 초기 Random Forest 결과 기록 |
| 입력 데이터 | `rf_0417_result*.csv`, `rf_0417_resul*.csv`, `READ_ME.txt` |
| 수행 내용 | fold 수와 SMOTE / ADASYN 조합 비교 |
| 저장소 역할 | 기준선 결과 보관용 저장소 |

## 주요 결과

| Setting | Accuracy | F1-score | Recall |
| --- | ---: | ---: | ---: |
| `3-fold + ADASYN` | 0.642 | 0.540 | 0.654 |
| `3-fold + SMOTE` | 0.617 | 0.508 | 0.615 |
| `10-fold + ADASYN` | 0.580 | 0.393 | 0.423 |

보존된 결과를 기준으로 보면 `3-fold + ADASYN` 설정이 가장 높은 F1-score를 기록하였으며, 초기 기준선 단계에서는 sampling 전략이 성능 차이에 큰 영향을 주었다.

## 주요 파일

| 파일 | 설명 |
| --- | --- |
| `READ_ME.txt` | 결과 파일과 실험 설정 대응 메모 |
| `rf_0417_result_10fold.csv` | 10-fold 결과 |
| `rf_0417_result_3fold_adasyn.csv` | 최고 성능 보존 결과 |
