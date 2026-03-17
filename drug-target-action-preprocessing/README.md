# Drug Target Action Preprocessing

본 저장소는 후속 모델링 실험에 사용된 입력 데이터를 정리하기 위한 전처리 단계이다. 약물명 정리, DrugBank ID 매칭, target-action pivot 생성, `SMILES` 보정 작업을 수행하였다.

## English Summary

This repository represents the preprocessing stage used to build the downstream modeling inputs. It covers drug name normalization, DrugBank ID alignment, target-action pivot generation, and `SMILES` cleanup.

## 연구 개요

| 항목 | 내용 |
| --- | --- |
| 연구 초점 | 입력 데이터 정합화 및 target-action matrix 생성 |
| 입력 데이터 | `drug_target_v2 (1).csv`, `structure_links.csv`, `smile_data_0627_1.csv` |
| 수행 내용 | 문자열 정규화, 병합, pivot generation |
| 저장소 역할 | 통합 데이터셋 구축의 기반 단계 |

## 주요 파일

| 파일 | 설명 |
| --- | --- |
| `data_preprocessing.py` | DrugBank ID 연결 및 `SMILES` 정리 |
| `merge.py` | 보조 병합 스크립트 |
| `target_action_make.py` | target-action pivot 생성 |
