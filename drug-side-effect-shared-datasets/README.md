# Drug Side Effect Shared Datasets

이 저장소는 여러 실험 폴더에서 공통으로 사용한 데이터셋을 모아둔 기준 데이터 저장소이다. 실제 모델링 레포들이 참조하는 입력 파일을 중앙에서 관리하는 역할을 한다.

## English Summary

This repository serves as the canonical shared-data layer for the project. It stores the common datasets referenced by multiple modeling repositories.

## 연구 개요

| 항목 | 내용 |
| --- | --- |
| 연구 초점 | 공통 입력 데이터 관리 |
| 입력 데이터 | `dataset_0629.csv`, `dataset_0722.csv`, `datasets_smiles*.csv`, `target_pivot.csv`, `structure_links.csv` |
| 수행 내용 | 전처리 결과와 통합 데이터셋을 공통 계층으로 정리 |
| 저장소 역할 | 다른 레포들을 연결하는 데이터 중심 저장소 |

## 주요 파일

| 파일 | 설명 |
| --- | --- |
| `dataset_0629.csv` | 0629 계열 모델링에 사용된 통합 데이터셋 |
| `dataset_0722.csv` | 0722 계열 실험 데이터셋 |
| `datasets_smiles_approval.csv` | 구조 및 승인일 정보 중심 데이터셋 |
| `datasets_target_smiles_approval.csv` | 구조 + target + 승인일 데이터셋 |
