# Drug Metadata Collection Tools

본 저장소는 외부 메타데이터를 보완하기 위해 작성한 보조 수집 스크립트를 모아둔 것이다. Wikipedia, 외부 약물 정보 서비스, Selenium 기반 자동화를 이용해 누락 정보를 보완하려는 시도를 포함한다.

## English Summary

This repository contains support scripts for collecting external metadata. It includes attempts to complement missing information using Wikipedia, pharmacology-related tools, and Selenium-based automation.

## 연구 개요

| 항목 | 내용 |
| --- | --- |
| 연구 초점 | 메타데이터 수집 및 보완 |
| 입력 데이터 | `slimes_data_v2.csv`, `smile_data_v3.csv`, 외부 웹 자원 |
| 수행 내용 | API 조회, 웹 검색, Selenium 자동화 |
| 저장소 역할 | 모델링 이전 보조 정보 수집 단계 |

## 주요 파일

| 파일 | 설명 |
| --- | --- |
| `api.py` | Wikipedia 존재 여부 및 URL 수집 |
| `crolling.py` | 승인 연도 등 메타데이터 수집 |
| `test.py` | Selenium 기반 자동화 테스트 |
