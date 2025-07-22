# ner_processor.py

import os
import pandas as pd
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline
import numpy as np
import re
import logging
import json
from tqdm import tqdm
from fuzzywuzzy import process

# 모듈화된 설정 파일 import
import config

# --- 로거 설정 ---
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------- #
# --- 원본 스크립트의 모든 전처리/파싱/헬퍼 함수들 ---
# (이하 함수들은 제공해주신 코드와 동일합니다)
# ---------------------------------------------------------------- #

def preprocess_report_text(text):
    if not isinstance(text, str): return ""
    text = re.sub(r'Moraxella_sg_Branhamella', 'Moraxella', text, flags=re.IGNORECASE)
    text = re.sub(r'\(Pseudo\.\)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s*–\s*MALDI-TOF.*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s*by\s+biotyper.*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'동정됨$', '', text.strip()).strip()
    text = text.strip('":- \t')
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'\bP\d{1,2}\s*:?', '', text)
    return text


def prettify_resistance_phrase(phrase):
    phrase = re.sub(r'(ESBL|AmpC|MRSA|MSSA|VRE|CRE)[ ]?([Pp]os|[Nn]eg|[\+\-])', r'\1 \2', phrase)
    phrase = re.sub(r'([Pp]os|[Nn]eg)[ ]?(\()', r'\1 \2', phrase)
    phrase = re.sub(r'([ESBL|AmpC|MRSA|MSSA|VRE|CRE])[\s]*([\+\-])', r'\1 \2', phrase)
    phrase = re.sub(r'\s*\(\s*', ' (', phrase)
    phrase = re.sub(r'\s*\)\s*', ')', phrase)
    phrase = phrase.strip()
    return phrase


def split_multi_organism_reports(report_text):
    if not isinstance(report_text, str) or '동정결과:' not in report_text:
        return [report_text]
    delimiter = "동정결과:"
    chunks = report_text.split(delimiter)
    split_reports = [delimiter + chunk.strip() for chunk in chunks[1:] if chunk.strip()]
    prefix_content = chunks[0].strip()
    if prefix_content and split_reports:
        split_reports[0] = prefix_content + "\n" + split_reports[0]
    return split_reports if split_reports else [report_text]


def extract_and_remove_resistances(report_text, resistance_dict):
    found_resistances = set()
    cleaned_text = report_text
    special_pattern = re.compile(
        r'Inducible\s+Clindamycin\s+Resistance\s+[Pp]os\s*\(\+\)|Inducible\s+Clindamycin\s+Resistance\s+[Nn]eg\s*\(\-\)',
        re.IGNORECASE)
    matches = special_pattern.findall(cleaned_text)
    for m in matches:
        found_resistances.add(m.strip())
        cleaned_text = cleaned_text.replace(m, '')
    patterns = ['ESBL', 'AmpC', 'MRSA', 'MSSA', 'VRE', 'CRE']
    for key in patterns:
        matches = re.findall(rf'{key}[^,\n;]*', cleaned_text, re.IGNORECASE)
        for m in matches:
            phrase = m.strip(" ;,:\n\r\t")
            if phrase.lower().startswith(key.lower()) and len(phrase) > len(key):
                found_resistances.add(phrase)
                cleaned_text = cleaned_text.replace(m, '')
    for key, value in sorted(resistance_dict.items(), key=lambda item: len(item[1]), reverse=True):
        if value and value in cleaned_text:
            found_resistances.add(value)
            cleaned_text = cleaned_text.replace(value, '')
    for key, value in resistance_dict.items():
        if key in cleaned_text:
            found_resistances.add(value)
            cleaned_text = cleaned_text.replace(key, '')
    found_resistances_cleaned = [' '.join(x.split()) for x in found_resistances]
    cleaned_text = re.sub(r'\s{2,}', ' ', cleaned_text).strip()
    return cleaned_text, found_resistances_cleaned


def parse_antibiotic_block(block_text):
    results = []
    for line in block_text.split('\n'):
        match = re.match(r'^(.+?)\s{2,}([<>=]*\s*\d+\.?\d*(?:\s*\([^)]+\))?)\s+\(([SIR\+\-])\)', line.strip())
        if match:
            abx_name = match.group(1).strip()
            mic_raw = match.group(2)
            mic = re.sub(r'\s*\([^)]*\)', '', mic_raw)
            mic = mic.replace(' ', '').strip()
            interp = match.group(3)
            results.append({'name': abx_name, 'mic': mic, 'interpretation': interp})
    return results


def extract_antibiotic_block(report_text):
    pattern = re.compile(
        r'항생제\s*감수성결과\s*\n[-]+\n항생제명\s+결과값\s+판정\n[-]+\n(.*?)(?:-+\n|$)', re.DOTALL)
    match = pattern.search(report_text)
    if match:
        return match.group(1)
    return ''


def clean_model_output(text_input):
    if not isinstance(text_input, str): return text_input
    cleaned_text = re.sub(r'^[ \ ]+|[ \ ,]+$', '', text_input)
    return cleaned_text.strip()


def parse_date_string(date_str):
    if not isinstance(date_str, str) or str(date_str).lower() == 'null' or pd.isna(date_str): return None
    try:
        if '오전' in date_str or '오후' in date_str:
            dt_str = date_str.replace('오전', 'AM').replace('오후', 'PM')
            return pd.to_datetime(dt_str, format='%Y-%m-%d %p %I:%M:%S')
        return pd.to_datetime(date_str)
    except (ValueError, TypeError):
        return None


def extract_with_regex(report_text):
    comment = None
    final_report_dt = None
    datetime_matches = re.findall(r'결과보고시간\s*:?\s*(\d{4}-\d{2}-\d{2}\s*오[전후]\s*\d{1,2}:\d{2}:\d{2})', report_text)
    if datetime_matches:
        last_dt_str = datetime_matches[-1].replace('오전', 'AM').replace('오후', 'PM')
        try:
            final_report_dt = pd.to_datetime(last_dt_str, format='%Y-%m-%d %p %I:%M:%S')
        except ValueError:
            pass
    explicit_comment_match = re.search(r'COMMENT\s*:\s*(.*)', report_text, re.DOTALL)
    if explicit_comment_match:
        comment_text = explicit_comment_match.group(1)
        final_report_marker = re.search(r'\[최종보고\]|결과보고시간', comment_text)
        if final_report_marker:
            comment = comment_text[:final_report_marker.start()].strip()
        else:
            comment = comment_text.strip()
    else:
        abx_block_match = re.search(
            r'항생제\s*감수성결과\s*\n[-]+\n항생제명\s+결과값\s+판정\n[-]+\n(?:.*\n)*?.*(?:-+\n|\Z)',
            report_text, re.DOTALL)
        if abx_block_match:
            start_pos = abx_block_match.end()
            text_after_abx = report_text[start_pos:]
            final_report_marker_match = re.search(r'\[최종보고\]|결과보고시간', text_after_abx)
            if final_report_marker_match:
                comment = text_after_abx[:final_report_marker_match.start()].strip()
            else:
                comment = text_after_abx.strip()
    if comment and not comment.strip():
        comment = None
    return comment, final_report_dt


def group_entities(ner_results):
    grouped_entities = []
    current_entity = None
    for res in ner_results:
        entity_tag = res['entity']
        clean_tag = entity_tag[2:] if entity_tag.startswith(('B-', 'I-')) else entity_tag
        is_new_entity = (
                    entity_tag.startswith('B-') or (current_entity and clean_tag != current_entity['entity_group']))
        if is_new_entity:
            if current_entity: grouped_entities.append(current_entity)
            current_entity = {'word': res['word'], 'entity_group': clean_tag, 'scores': [res['score']]}
        elif current_entity and clean_tag == current_entity['entity_group']:
            current_entity['word'] += ' ' + res['word']
            current_entity['scores'].append(res['score'])
        else:
            if current_entity: grouped_entities.append(current_entity)
            current_entity = None
    if current_entity: grouped_entities.append(current_entity)
    for entity in grouped_entities:
        clean_word = entity['word'].replace(' ', ' ').replace('\u00A0', ' ')
        entity['word'] = clean_word.strip()
        entity['score'] = np.mean(entity['scores'])
    return grouped_entities


def structure_multiple_results(grouped_entities):
    all_organisms_data = []
    current_organism = None
    for entity in grouped_entities:
        entity_group = entity['entity_group']
        if entity_group == 'ORGANISM':
            if current_organism:
                all_organisms_data.append(current_organism)
            current_organism = {"organism_name": entity['word'], "quantity_details": None, "key_resistance": None,
                                "antibiotic_results": []}
        elif current_organism:
            if entity_group == 'QUANTITY':
                current_organism['quantity_details'] = entity['word']
            elif entity_group == 'RESISTANCE':
                current_organism['key_resistance'] = entity['word']
            elif entity_group == 'ANTIBIOTIC':
                abx_text = entity['word']
                mic_match = re.search(r'((?:>=|<=|>|<|=|≤|≥)?\s*\d+\.?\d*)', abx_text)
                mic = mic_match.group(1) if mic_match else ''
                mic = re.sub(r'\(.*?\)', '', mic).replace(' ', '').strip()
                interp_match = re.search(r'\(([SIR\+\-])\)', abx_text)
                interp = interp_match.group(1) if interp_match else None
                name = re.sub(r'\s*\([^)]+\)', '', abx_text)
                name = re.sub(r'(>=|<=|>|<|=|≤|≥)?\s*\d+\.?\d*', '', name).strip()
                current_organism['antibiotic_results'].append({'name': name, 'mic': mic, 'interpretation': interp})
            elif entity_group == 'MIC' and current_organism['antibiotic_results'] and not \
            current_organism['antibiotic_results'][-1]['mic']:
                current_organism['antibiotic_results'][-1]['mic'] = entity['word']
            elif entity_group == 'INTERPRETATION' and current_organism['antibiotic_results']:
                current_organism['antibiotic_results'][-1]['interpretation'] = entity['word']
    if current_organism:
        all_organisms_data.append(current_organism)
    if all_organisms_data:
        return [{"result_type": "병원균 검출", "identified_organisms": all_organisms_data}]
    return []


def step1_rule_based_parser(report_text):
    lower_text = report_text.lower()
    if any(keyword in lower_text for keyword in
           ['동정불가', '동정 불가', 'identification fail', 'unable to identify', 'insignificant growth', 'gp rods', 'gpr',
            'gnr']):
        return [{"result_type": "동정 불가/미미한 성장", "identified_organisms": []}]
    if any(keyword in lower_text for keyword in
           ['no growth', 'no isolation', 'not found', 'no isolated', 'not isolated', 'no salmonella']):
        return [{"result_type": "균 없음", "identified_organisms": []}]
    if 'mixed growth' in lower_text or 'contamination' in lower_text:
        return [{"result_type": "혼합/오염균총", "identified_organisms": []}]
    if 'normal flora' in lower_text or 'nomal flora' in lower_text:
        return [{"result_type": "정상균총", "identified_organisms": []}]
    if not re.search(r'동정결과[:：]', report_text):
        if re.search(r'Time To Positivity|결과보고시간|TTP', report_text, re.IGNORECASE):
            return [{"result_type": "기타 (TTP 등 비정형 보고)", "identified_organisms": []}]
    if re.fullmatch(r'[\s\d\[\]최종보고:-]*', report_text.strip()):
        return [{"result_type": "기타/확인필요", "identified_organisms": []}]
    return None


def step2_local_ner_parser(ner_pipeline, report_text):
    ner_results = ner_pipeline(report_text)
    if not ner_results: return None
    grouped_results = group_entities(ner_results)
    return structure_multiple_results(grouped_results) if grouped_results else None


def step3_validate_and_correct(structured_data_list, dict_choices, interp_dict):
    if not structured_data_list or not structured_data_list[0].get('identified_organisms'):
        return None
    validated_organism_list = []
    valid_interp_values = set(interp_dict.keys()) | set(interp_dict.values())
    for organism in structured_data_list[0]['identified_organisms']:
        original_organism_name = clean_model_output(organism.get('organism_name'))
        if not original_organism_name: continue
        best_match, score = process.extractOne(original_organism_name, dict_choices['organism'])
        if score < config.SIMILARITY_THRESHOLD:  # 설정 파일 값 사용
            raise ValueError(f"유효하지 않은 동정균주명: '{original_organism_name}' (유사도: {score})")
        corrected_organism_name = best_match
        validated_antibiotics = []
        for abx in organism.get('antibiotic_results', []):
            original_abx_name = clean_model_output(abx.get('name'))
            original_interp = clean_model_output(abx.get('interpretation'))
            if not original_abx_name: continue
            best_abx, abx_score = process.extractOne(original_abx_name, dict_choices['antibiotic'])
            if abx_score < config.SIMILARITY_THRESHOLD:  # 설정 파일 값 사용
                raise ValueError(f"유효하지 않은 항생제명: '{original_abx_name}' (유사도: {abx_score})")
            corrected_interp = None
            if original_interp:
                if original_interp in valid_interp_values:
                    corrected_interp = interp_dict.get(original_interp, original_interp)
                else:
                    raise ValueError(f"사전에 없는 유효하지 않은 판정: '{original_interp}'")
            validated_antibiotics.append({'name': best_abx, 'mic': abx.get('mic'), 'interpretation': corrected_interp})
        validated_organism_list.append(
            {"organism_name": corrected_organism_name, "quantity_details": organism.get('quantity_details'),
             "key_resistance": organism.get('key_resistance'), "antibiotic_results": validated_antibiotics})
    if not validated_organism_list: return None
    return [{"result_type": "병원균 검출 (AI-검증)", "identified_organisms": validated_organism_list}]


def step4_dictionary_parser(report_text, organism_dictionary):
    found_organisms, negation_words = [], ['않음', '없음', '제외', '의심']
    lower_report = report_text.lower()
    pattern = re.search(r'동정결과[:：]?\s*([A-Za-z\s]+)[,，]', report_text) or re.search(r'동정결과[:：]?\s*([A-Za-z\s]+)',
                                                                                    report_text)
    if pattern:
        best_match, score = process.extractOne(pattern.group(1).strip(), list(organism_dictionary.values()))
        if score >= 80:
            found_organisms.append({"organism_name": best_match, "quantity_details": None, "key_resistance": None,
                                    "antibiotic_results": []})
    for korean_term, scientific_name in organism_dictionary.items():
        if korean_term.lower() in lower_report or scientific_name.lower() in lower_report:
            start_index = lower_report.find(
                korean_term.lower() if korean_term.lower() in lower_report else scientific_name.lower())
            if any(neg_word in lower_report[max(0, start_index - 15):start_index + 15] for neg_word in negation_words):
                continue
            if not any(org['organism_name'] == scientific_name for org in found_organisms):
                found_organisms.append(
                    {"organism_name": scientific_name, "quantity_details": None, "key_resistance": None,
                     "antibiotic_results": []})
    if found_organisms:
        return [{"result_type": "병원균 검출 (사전 fallback)", "identified_organisms": found_organisms}]
    return None


def simplify_resistance(resistance_str):
    if not isinstance(resistance_str, str): return resistance_str
    res_list = [r.strip() for r in resistance_str.split(',')]
    new_list = [r if 'MRSA' not in r else 'MRSA' for r in res_list]
    if set(new_list) == {'MRSA'}: return 'MRSA'
    return ', '.join(sorted(list(set(x for x in new_list if x))))


SPECIAL_COMMENT_HEAD = "Salmonella와 Shigella는 1,2세대 cephalosporin과 aminoglycoside에 대하여 임상적으로 효과가 없습니다."


# ---------------------------------------------------------------- #
# --- 메인 처리 함수 ---
# ---------------------------------------------------------------- #

def process_dataframe(df: pd.DataFrame):
    """
    전체 데이터프레임을 입력받아 처리하고, 3개의 결과 데이터프레임을 반환합니다.
    (기존 main_colab 함수의 로직을 여기에 통합)
    """
    logger.info("Starting dataframe processing...")

    # --- 모델 및 사전 로드 ---
    try:
        ner_pipeline = pipeline("ner", model=config.LOCAL_MODEL_PATH, tokenizer=config.LOCAL_MODEL_PATH,
                                device=0 if torch.cuda.is_available() else -1)
        logger.info(f"Local NER model loaded from '{config.LOCAL_MODEL_PATH}'.")
    except Exception as e:
        logger.error(f"Failed to load NER model: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    try:
        with open(config.MEDICAL_DICT_PATH, 'r', encoding='utf-8') as f:
            full_medical_dict = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load medical dictionary: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    organism_dict = full_medical_dict.get("ORGANISM", {})
    interp_dict = full_medical_dict.get("INTERPRETATION", {})
    resistance_dict = full_medical_dict.get("RESISTANCE", {})
    dict_choices = {
        'organism': list(organism_dict.values()),
        'antibiotic': list(full_medical_dict.get("ANTIBIOTIC", {}).values()),
    }

    processed_data_rows, antibiotic_data_rows, failed_data_rows = [], [], []
    next_result_id = 1

    for index, row in tqdm(df.iterrows(), total=len(df), desc="DataFrame Processing"):
        raw_report_text = str(row.get('검사결과', ''))
        report_texts_list = split_multi_organism_reports(raw_report_text)

        for single_report_text in report_texts_list:
            structured_data_list = None
            try:
                temp_text, found_resistances = extract_and_remove_resistances(single_report_text, resistance_dict)
                report_text = preprocess_report_text(temp_text)

                if '미생물 배양 불가' in report_text:
                    structured_data_list = [{"result_type": "기타 (TTP 등 비정형 보고)", "identified_organisms": []}]
                elif not report_text.strip() or report_text.lower() == 'nan':
                    structured_data_list = [{"result_type": "균 없음 (빈 셀)", "identified_organisms": []}]

                if not structured_data_list:
                    structured_data_list = step1_rule_based_parser(report_text)

                if not structured_data_list:
                    ai_results = step2_local_ner_parser(ner_pipeline, report_text)
                    if ai_results:
                        try:
                            structured_data_list = step3_validate_and_correct(ai_results, dict_choices, interp_dict)
                        except ValueError:
                            structured_data_list = step4_dictionary_parser(report_text, organism_dict)

                if not structured_data_list:
                    structured_data_list = step4_dictionary_parser(report_text, organism_dict)

                if not structured_data_list:
                    raise ValueError("모든 파싱 및 검증 단계에서 처리 실패")

                for result in structured_data_list:
                    organisms = result.get('identified_organisms', [])
                    comment_text, final_report_time = extract_with_regex(single_report_text)

                    if not organisms:
                        processed_data_rows.append({
                            '결과_ID': next_result_id, '원본_내원번호': row.get('내원번호'), '원본_환자번호': row.get('환자번호'),
                            '성별': row.get('성별'), '생년월일': parse_date_string(row.get('생년월일')),
                            '입원일': parse_date_string(row.get('입원일')), '검사시행일시': parse_date_string(row.get('검사시행일시')),
                            '결과유형': result.get('result_type'), '동정균주명': None, '균_정량_상세': None,
                            '주요내성_특징': None, '코멘트': comment_text, '최종보고일시': final_report_time,
                            '검사명': row.get('검사명'), '검체명_주검체': row.get('검체명_주검체)')
                        })
                        next_result_id += 1
                    else:
                        for organism in organisms:
                            ai_resistance = organism.get('key_resistance')
                            if ai_resistance: found_resistances.add(ai_resistance)
                            all_resistances = list(found_resistances)
                            organism['key_resistance'] = ', '.join(
                                sorted(list(set(all_resistances)))) if all_resistances else None
                            key_resistance_value = clean_model_output(organism.get('key_resistance'))

                            if comment_text:
                                simple_comment = re.sub(r'\s+', '', comment_text)
                                simple_special = re.sub(r'\s+', '', SPECIAL_COMMENT_HEAD)
                                if simple_special in simple_comment:
                                    key_resistance_value = SPECIAL_COMMENT_HEAD

                            abx_block = extract_antibiotic_block(single_report_text)
                            if abx_block:
                                abx_results = parse_antibiotic_block(abx_block)
                                for abx in abx_results:
                                    antibiotic_data_rows.append({
                                        '항생제결과_ID': len(antibiotic_data_rows) + 1, '결과_ID': next_result_id,
                                        '항생제명': abx['name'], 'MIC_결과값': abx['mic'], '판정': abx['interpretation']
                                    })

                            processed_data_rows.append({
                                '결과_ID': next_result_id, '원본_내원번호': row.get('내원번호'), '원본_환자번호': row.get('환자번호'),
                                '성별': row.get('성별'), '생년월일': parse_date_string(row.get('생년월일')),
                                '입원일': parse_date_string(row.get('입원일')),
                                '검사시행일시': parse_date_string(row.get('검사시행일시')),
                                '결과유형': result.get('result_type'),
                                '동정균주명': clean_model_output(organism.get('organism_name')),
                                '균_정량_상세': clean_model_output(organism.get('quantity_details')),
                                '주요내성_특징': key_resistance_value,
                                '코멘트': comment_text, '최종보고일시': final_report_time, '검사명': row.get('검사명'),
                                '검체명_주검체': row.get('검체명_주검체)')
                            })
                            for abx in organism.get('antibiotic_results', []):
                                antibiotic_data_rows.append({
                                    '항생제결과_ID': len(antibiotic_data_rows) + 1, '결과_ID': next_result_id,
                                    '항생제명': clean_model_output(abx.get('name')),
                                    'MIC_결과값': clean_model_output(abx.get('mic')),
                                    '판정': clean_model_output(abx.get('interpretation'))
                                })
                            next_result_id += 1

            except Exception as e:
                failed_data_rows.append({
                    '실패_ID': len(failed_data_rows) + 1, '원본_내원번호': row.get('내원번호'),
                    '원본_환자번호': row.get('환자번호'), '실패사유': str(e), '원본_검사결과': single_report_text
                })

    # --- 최종 데이터 정리 ---
    for row in processed_data_rows:
        row['주요내성_특징'] = simplify_resistance(row['주요내성_특징'])

    processed_df = pd.DataFrame(processed_data_rows)
    antibiotic_df = pd.DataFrame(antibiotic_data_rows)

    if not antibiotic_df.empty:
        antibiotic_df['MIC_결과값'] = (
            antibiotic_df['MIC_결과값'].astype(str)
            .str.replace(r'\s+', '', regex=True)
            .str.extract(r'([<>=]*\d+\.?\d*)', expand=False)
        )

    failed_df = pd.DataFrame(failed_data_rows)

    logger.info("DataFrame processing finished.")
    return processed_df, antibiotic_df, failed_df