import csv
import os
import zipfile
import json
import re
from collections import Counter
from bs4 import BeautifulSoup, NavigableString
from google import genai
from google.genai import types
import xml.etree.ElementTree as ET
from tqdm import tqdm
import time
import datetime
from xml.dom import minidom
from PIL import Image, ImageDraw, ImageFont
import io
from google.api_core.exceptions import InvalidArgument
import shutil
import logging
import colorama
import posixpath
import concurrent.futures
import copy
import uuid
import traceback

# ── google.genai GenerativeModel 호환 래퍼 ───────────────────────────
class GeminiModel:
    """google.genai Client 기반 래퍼 (GenerativeModel 인터페이스 호환)"""
    def __init__(self, client, model_name, safety_settings=None, generation_config=None):
        self.client = client
        self.model_name = model_name
        self.safety_settings = safety_settings or []
        self.generation_config = generation_config or {}

    def generate_content(self, prompt, stream=False):
        cfg = types.GenerateContentConfig(
            temperature=self.generation_config.get('temperature'),
            top_p=self.generation_config.get('top_p'),
            top_k=self.generation_config.get('top_k'),
            safety_settings=self.safety_settings,
        )
        if stream:
            return self.client.models.generate_content_stream(
                model=self.model_name,
                contents=prompt,
                config=cfg,
            )
        else:
            return self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=cfg,
            )


# ── Gemini safety settings ───────────────────────────
safety_settings = [
    types.SafetySetting(
        category="HARM_CATEGORY_HARASSMENT",
        threshold="BLOCK_NONE"
    ),
    types.SafetySetting(
        category="HARM_CATEGORY_HATE_SPEECH",
        threshold="BLOCK_NONE"
    ),
    types.SafetySetting(
        category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
        threshold="BLOCK_NONE"
    ),
    types.SafetySetting(
        category="HARM_CATEGORY_DANGEROUS_CONTENT",
        threshold="BLOCK_NONE"
    ),
]                 
                    
# --- 0. 프롬프트 설정 (전역 변수) ---

base_prompt_instructions = """
"""


base_prompt_text = """
############### 이전 문맥 ###############

  prev_context:
  {prev_context}

아래는 당신이 번역해야 할 current text: 부분입니다. current text: 이후의 내용들을 위의 지침에 따라 번역해주세요.
위의 prev_context:~ 부분은 번역 시 문맥을 참고만 하고 최종 출력물에 *절대* 포함하지 않습니다.

############### 번역 시작 ###############

  current text:
  {current_text}
"""


# --- 0-2. 2차 번역 프롬프트 설정 (전역 변수) 한국어로 번역된 문장을 다시 가다듬는 작업 진행 ---

SECOND_TRANSLATION_PROMPT_INSTRUCTIONS = """


"""


SECOND_TRANSLATION_PROMPT_TEXT = """
**번역 출력 시 주의사항: 2차 번역이 완료된 내용만을 출력합니다. 추가적인 설명이나 질문, 부가 사항은 출력하지 않습니다.

다음은 수정해야 할 한국어 텍스트입니다.

############### 수정 시작 ###############

{current_text}
"""

# --- 0-3. 목차 번역 사전 (전역 변수) ---
NAV_TRANSLATIONS = {
    "表紙": "표지", "奥付": "판권", "目次": "목차", "もくじ": "목차",
    "あとがき": "후기", "CONTENTS": "목차", "INDEX": "색인", "本編": "본문"
    # 필요시 다른 항목 추가
}


# --- 1. 설정 및 유틸리티 ---
def print_colored(text, color=colorama.Fore.RESET, style=colorama.Style.RESET_ALL):
    print(f"{style}{color}{text}{colorama.Style.RESET_ALL}")
    

def count_non_whitespace(text):
    """주어진 텍스트에서 공백 문자를 제외한 글자 수를 반환합니다."""
    if not text:
        return 0
    # 모든 종류의 공백(\s)을 제거하고 길이를 계산
    pure_text = re.sub(r'\s+', '', text)
    return len(pure_text)

    
def get_api_key_and_params():
    """settings.csv 파일에서 API 키 및 파라미터들을 가져옵니다."""
    settings_file = "settings.csv"
    gemini_api_key_param_name = "Gemini API key"
    temperature_param_name = "temperature"
    top_p_param_name = "top_p"
    top_k_param_name = "top_k"
    text_block_size_param_name = "text block size"
    previous_context_number_param_name = "previous context number"
    retranslate_max_retries_param_name = "retranslate max retries"
    second_translation_param_name = "2nd translation"
    number_of_parallel_processing_param_name = "number of parallel processing"
    cover_image_modify_param_name = "cover image modify"
    cover_text_position_param_name = "cover text position"
    cover_text_param_name = "cover text"
    font_param_name = "font"
    font_size_param_name = "font size"
    font_color_param_name = "font color"
    background_color_param_name = "background color"
    delete_temp_files_param_name = "Delete temp files"
    ridi_version_param_name = "RIDI_VERSION"

    # 기본값 (default values)
    default_temperature = 0.9
    default_top_p = 0.9
    default_top_k = 40
    default_text_block_size = 6500
    default_previous_context_number = 5
    default_retranslate_max_retries = 2
    default_second_translation = 2
    default_num_parallel = 20
    default_cover_image_modify = 2
    default_cover_text_position = 1
    default_cover_text = "AI 번역본"
    default_font = "RIDIBatang.otf"
    default_font_size = 50
    default_font_color = "FFFFFF"
    default_background_color = "FF0000"
    default_delete_temp_files = 1
    default_ridi_version = 2

    default_values = {
        gemini_api_key_param_name: "",
        temperature_param_name: default_temperature,
        top_p_param_name: default_top_p,
        top_k_param_name: default_top_k,
        text_block_size_param_name: default_text_block_size,
        previous_context_number_param_name: default_previous_context_number,
        retranslate_max_retries_param_name: default_retranslate_max_retries,
        second_translation_param_name: default_second_translation,
        number_of_parallel_processing_param_name: default_num_parallel,
        cover_image_modify_param_name: default_cover_image_modify,
        cover_text_position_param_name: default_cover_text_position,
        cover_text_param_name: default_cover_text,
        font_param_name: default_font,
        font_size_param_name: default_font_size,
        font_color_param_name: default_font_color,
        background_color_param_name: default_background_color,
        delete_temp_files_param_name: default_delete_temp_files,
        ridi_version_param_name: default_ridi_version
    }

    def create_settings_file(filename):
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["parameter", "value", "remark"])
            writer.writerow([gemini_api_key_param_name, "", "Gemini API Key"])
            writer.writerow([temperature_param_name, default_temperature, "Temperature (0.0 ~ 1.0 default=0.9)"])
            writer.writerow([top_p_param_name, default_top_p, "Top-p (0.0 ~ 1.0 default=0.9)"])
            writer.writerow([top_k_param_name, default_top_k, "Top-k (1-40 default=40)"])
            writer.writerow([text_block_size_param_name, default_text_block_size, "Number of character (글자 수 * 0.75 = 토큰 수 추정. default=6500, max=80000)"])
            writer.writerow([previous_context_number_param_name, default_previous_context_number, "프롬프트 참고용 이전 문맥 줄 수 (default=5)"])
            writer.writerow([retranslate_max_retries_param_name, default_retranslate_max_retries, "Max Retranslate Retries (0-5, default=2)"])
            writer.writerow([second_translation_param_name, default_second_translation, "2차 번역 수행 여부 (1: 수행, 2: 미수행, default=2)"])
            writer.writerow([number_of_parallel_processing_param_name, default_num_parallel, "병렬 번역 텍스트 수 (1 이상, default=20)"])
            writer.writerow([cover_image_modify_param_name, default_cover_image_modify, "Cover Image Modify (1: 사용, 2: 미사용 default=2)"])
            writer.writerow([cover_text_position_param_name, default_cover_text_position, "Cover Text Position (1: TL, 2: TR, 3: BL, 4: BR default=1)"])
            writer.writerow([cover_text_param_name, default_cover_text, "Cover Text default=AI 번역본"])
            writer.writerow([font_param_name, default_font, "Font filename (default=RIDIBatang.otf)"])
            writer.writerow([font_size_param_name, default_font_size, "Font Size (default=50)"])
            writer.writerow([font_color_param_name, default_font_color, "Font Color (RGB hex, default=FFFFFF)"])
            writer.writerow([background_color_param_name, default_background_color, "Background Color (RGB hex, default=FF0000)"])
            writer.writerow([delete_temp_files_param_name, default_delete_temp_files, "번역 임시 파일 삭제 (1: 삭제, 2: 유지 default=1)"])
            writer.writerow([ridi_version_param_name, default_ridi_version, "RIDI 버전 추가 생성 (1: 생성, 2: 미생성, default=2)"])
        print(f"{filename} 파일이 생성되었습니다.")

    def get_api_key_from_user():
        while True:
            api_key = input("Gemini API 키를 입력하세요: ")
            if api_key.strip(): return api_key
            print("API 키는 반드시 입력해야 합니다.")

    def get_float_param_from_user(param_name, default_value, min_val, max_val):
        while True:
            try:
                value_str = input(f"{param_name} 값 입력 (기본값: {default_value}, 범위: {min_val}~{max_val}): ")
                if not value_str: return default_value
                value = float(value_str)
                if min_val <= value <= max_val: return value
                else: print(f"유효 범위({min_val}~{max_val}) 벗어남.")
            except ValueError: print("숫자 입력 필요.")

    def get_int_param_from_user(param_name, default_value, min_val=None, max_val=None):
        while True:
            try:
                value_str = input(f"{param_name} 값 입력 (기본값: {default_value}" + (f", 범위: {min_val}~{max_val}" if min_val is not None and max_val is not None else "") + "): ")
                if not value_str: return default_value
                value = int(value_str)
                if min_val is not None and value < min_val: print(f"최소값 {min_val} 이상이어야 합니다."); continue
                if max_val is not None and value > max_val: print(f"최대값 {max_val} 이하여야 합니다."); continue
                return value
            except ValueError: print("정수 입력 필요.")
    # ---

    # save_params_to_file 함수 시그니처 및 내용 수정
    def save_params_to_file(filename, api_key, temperature, top_p, top_k, text_block_size, previous_context_number, retranslate_max_retries, second_translation, num_parallel, cover_image_modify, cover_text_position, cover_text, font, font_size, font_color, background_color, delete_temp_files, ridi_version):
        params_to_add = {
            gemini_api_key_param_name: [api_key, "Gemini API Key"],
            temperature_param_name: [temperature, f"Temperature (0.0 ~ 2.0 default={default_temperature})"],
            top_p_param_name: [top_p, f"Top-p (0.0 ~ 1.0 default={default_top_p})"],
            top_k_param_name: [top_k, f"Top-k (1-40 default={default_top_k})"],
            text_block_size_param_name: [text_block_size, f"Number of character (default={default_text_block_size}, max=80000)"],
            previous_context_number_param_name: [previous_context_number, f"프롬프트 참고용 이전 문맥 줄 수 (default={default_previous_context_number})"],
            retranslate_max_retries_param_name: [retranslate_max_retries, f"Max Retranslate Retries (0-5, default={default_retranslate_max_retries})"],
            second_translation_param_name: [second_translation, f"2차 번역 수행 여부 (1: 수행, 2: 미수행, default={default_second_translation})"],
            number_of_parallel_processing_param_name: [num_parallel, f"Number of Parallel Processing (병렬 번역 텍스트 수 (1 이상, default={default_num_parallel})"],
            cover_image_modify_param_name: [cover_image_modify, f"Cover Image Modify (1: 사용, 2: 미사용 default={default_cover_image_modify})"],
            cover_text_position_param_name: [cover_text_position, f"Cover Text Position (1: TL, 2: TR, 3: BL, 4: BR default={default_cover_text_position})"],
            cover_text_param_name: [cover_text, f"Cover Text default={default_cover_text}"],
            font_param_name: [font, f"Font filename (default={default_font})"],
            font_size_param_name: [font_size, f"Font Size (default={default_font_size})"],
            font_color_param_name: [font_color, f"Font Color (RGB hex, default={default_font_color})"],
            background_color_param_name: [background_color, f"Background Color (RGB hex, default={default_background_color})"],
            delete_temp_files_param_name: [delete_temp_files, f"번역 임시 파일 삭제 (1: 삭제, 2: 유지 default={default_delete_temp_files})"],
            ridi_version_param_name: [ridi_version, f"RIDI 버전 추가 생성 (1: 생성, 2: 미생성, default={default_ridi_version})"]
        }

        try:
            with open(filename, 'r', newline='', encoding='utf-8') as csvfile:
                original_rows = list(csv.reader(csvfile))
        except FileNotFoundError:
            original_rows = []

        rows = []
        header = ["parameter", "value", "remark"]
        existing_params = set()
        if original_rows and len(original_rows) > 1:
            for row in original_rows[1:]:
                if len(row) >= 2: 
                    param_name = row[0].strip()
                    if param_name in params_to_add:
                        row[1] = params_to_add[param_name][0]
                        if len(row) < 3 or row[2].strip() != params_to_add[param_name][1]:
                           if len(row) < 3: row.append("")
                           row[2] = params_to_add[param_name][1]
                        rows.append(row)
                        existing_params.add(param_name)
                    else:
                        rows.append(row)
                        existing_params.add(param_name)
                else:
                    print(f"Warning: settings.csv의 행 무시 (열 부족): {row}")

        file_changed = False
        for param_name, (value, remark) in params_to_add.items():
            if param_name not in existing_params:
                rows.append([param_name, value, remark])
                print(f"settings.csv 파일에 '{param_name}' 파라미터 추가됨.")
                file_changed = True

        rows.insert(0, header)

        if not file_changed: 
            if len(original_rows) != len(rows):
                file_changed = True
            else:
                min_len = min(len(rows), len(original_rows))
                for i in range(min_len):
                    if rows[i][:3] != original_rows[i][:3]:
                        file_changed = True
                        break

        if file_changed:
            try:
                with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerows(rows)
                logging.info(f"Saved updated settings to {filename}")
            except Exception as e:
                print_colored(f"Error saving settings to {filename}: {e}", colorama.Fore.RED)
                logging.error(f"Error saving settings to {filename}: {e}", exc_info=True)


    # --- 메인 로직 시작 ---
    if not os.path.exists(settings_file):
        # 파일 없을 때 생성 및 기본값 사용/저장
        create_settings_file(settings_file)
        api_key = get_api_key_from_user()
        temperature = default_values[temperature_param_name]
        top_p = default_values[top_p_param_name]
        top_k = default_values[top_k_param_name]
        text_block_size = default_values[text_block_size_param_name]
        previous_context_number = default_values[previous_context_number_param_name]
        retranslate_max_retries = default_values[retranslate_max_retries_param_name]
        second_translation = default_values[second_translation_param_name]
        num_parallel = default_values[number_of_parallel_processing_param_name]
        cover_image_modify = default_values[cover_image_modify_param_name]
        cover_text_position = default_values[cover_text_position_param_name]
        cover_text = default_values[cover_text_param_name]
        font = default_values[font_param_name]
        font_size = default_values[font_size_param_name]
        font_color = default_values[font_color_param_name]
        background_color = default_values[background_color_param_name]
        delete_temp_files = default_values[delete_temp_files_param_name]
        ridi_version = default_values[ridi_version_param_name]

        save_params_to_file(settings_file, api_key, temperature, top_p, top_k, text_block_size, previous_context_number, retranslate_max_retries, second_translation, num_parallel, cover_image_modify, cover_text_position, cover_text, font, font_size, font_color, background_color, delete_temp_files, ridi_version)
    else:
        # 파일 있을 때 읽기 및 검증
        try:
            with open(settings_file, 'r', newline='', encoding='utf-8') as csvfile:
                reader = csv.reader(csvfile)
                next(reader)
                params = {}
                for row in reader:
                    if len(row) >= 2: 
                        param_name = row[0].strip()
                        param_value = row[1].strip()
                        params[param_name] = param_value

            # 값 읽기 (get 사용 및 기본값 지정)
            api_key = params.get(gemini_api_key_param_name)
            temperature_str = params.get(temperature_param_name, str(default_temperature))
            top_p_str = params.get(top_p_param_name, str(default_top_p))
            top_k_str = params.get(top_k_param_name, str(default_top_k))
            text_block_size_str = params.get(text_block_size_param_name, str(default_text_block_size))
            previous_context_number_str = params.get(previous_context_number_param_name, str(default_previous_context_number))
            retranslate_max_retries_str = params.get(retranslate_max_retries_param_name, str(default_retranslate_max_retries))
            second_translation_str = params.get(second_translation_param_name, str(default_second_translation))
            num_parallel_str = params.get(number_of_parallel_processing_param_name, str(default_num_parallel))
            cover_image_modify_str = params.get(cover_image_modify_param_name, str(default_cover_image_modify))
            cover_text_position_str = params.get(cover_text_position_param_name, str(default_cover_text_position))
            cover_text = params.get(cover_text_param_name, default_cover_text)
            font = params.get(font_param_name, default_font)
            font_size_str = params.get(font_size_param_name, str(default_font_size))
            font_color = params.get(font_color_param_name, default_font_color)
            background_color = params.get(background_color_param_name, default_background_color)
            delete_temp_files_str = params.get(delete_temp_files_param_name, str(default_delete_temp_files))
            ridi_version_str = params.get(ridi_version_param_name, str(default_ridi_version))

            # API 키 검증 및 재입력
            if not api_key:
                print("settings.csv 파일에 API 키가 없습니다.")
                api_key = get_api_key_from_user()

            # 값 타입 변환 및 유효성 검증
            try:
                temperature = float(temperature_str)
                if not (0.0 <= temperature <= 2.0): raise ValueError
            except ValueError:
                print(f"settings.csv temperature 값 유효하지 않음. 기본값({default_temperature}) 사용."); temperature = default_temperature

            try:
                top_p = float(top_p_str)
                if not (0.0 <= top_p <= 1.0): raise ValueError
            except ValueError:
                print(f"settings.csv top_p 값 유효하지 않음. 기본값({default_top_p}) 사용."); top_p = default_top_p

            try:
                top_k = int(top_k_str)
                if top_k < 1: raise ValueError
            except ValueError:
                print(f"settings.csv top_k 값 유효하지 않음. 기본값({default_top_k}) 사용."); top_k = default_top_k

            try:
                text_block_size = int(text_block_size_str)
                if not (1 <= text_block_size <= 80000): raise ValueError
            except ValueError:
                print(f"settings.csv text block size 값 유효하지 않음. 기본값({default_text_block_size}) 사용."); text_block_size = default_text_block_size

            try:
                previous_context_number = int(previous_context_number_str)
                if previous_context_number < 0: raise ValueError
            except ValueError:
                print(f"settings.csv {previous_context_number_param_name} 값 유효하지 않음. 기본값({default_previous_context_number}) 사용."); previous_context_number = default_previous_context_number

            try:
                retranslate_max_retries = int(retranslate_max_retries_str)
                if not (0 <= retranslate_max_retries <= 5): raise ValueError
            except ValueError:
                print(f"settings.csv {retranslate_max_retries_param_name} 값 유효하지 않음. 기본값({default_retranslate_max_retries}) 사용."); retranslate_max_retries = default_retranslate_max_retries

            try:
                second_translation = int(second_translation_str)
                if second_translation not in (1, 2): raise ValueError
            except ValueError:
                print(f"settings.csv {second_translation_param_name} 값 유효하지 않음. 기본값({default_second_translation}) 사용."); second_translation = default_second_translation

            try:
                num_parallel = int(num_parallel_str)
                if num_parallel < 1: raise ValueError 
            except ValueError:
                print(f"settings.csv {number_of_parallel_processing_param_name} 값 유효하지 않음. 기본값({default_num_parallel}) 사용."); num_parallel = default_num_parallel

            try:
                cover_image_modify = int(cover_image_modify_str)
                if cover_image_modify not in (1, 2): raise ValueError
            except ValueError:
                print(f"settings.csv {cover_image_modify_param_name} 값 유효하지 않음. 기본값({default_cover_image_modify}) 사용."); cover_image_modify = default_cover_image_modify

            try:
                cover_text_position = int(cover_text_position_str)
                if not (1 <= cover_text_position <= 4): raise ValueError
            except ValueError:
                print(f"settings.csv {cover_text_position_param_name} 값 유효하지 않음. 기본값({default_cover_text_position}) 사용."); cover_text_position = default_cover_text_position

            try:
                font_size = int(font_size_str)
                if font_size < 1: raise ValueError
            except ValueError:
                print(f"settings.csv {font_size_param_name} 값 유효하지 않음. 기본값({default_font_size}) 사용."); font_size = default_font_size

            try:
                delete_temp_files = int(delete_temp_files_str)
                if delete_temp_files not in (1, 2): raise ValueError
            except ValueError:
                print(f"settings.csv {delete_temp_files_param_name} 값 유효하지 않음. 기본값({default_delete_temp_files}) 사용."); delete_temp_files = default_delete_temp_files
                
            try:
                ridi_version = int(ridi_version_str)
                if ridi_version not in (1, 2): raise ValueError
            except ValueError:
                print(f"settings.csv {ridi_version_param_name} 값 유효하지 않음 (1 또는 2 필요). 기본값({default_ridi_version}) 사용.")
                ridi_version = default_ridi_version

            # API 키 유효성 확인
            try:
                client = genai.Client(api_key=api_key)
            except Exception as e:
                print(f"API 키 유효하지 않음: {e}")
                api_key = get_api_key_from_user()

            save_params_to_file(settings_file, api_key, temperature, top_p, top_k, text_block_size, previous_context_number, retranslate_max_retries, second_translation, num_parallel, cover_image_modify, cover_text_position, cover_text, font, font_size, font_color, background_color, delete_temp_files, ridi_version)

        except Exception as e:
            # settings.csv 읽기 실패 시 전체 재설정
            print(f"settings.csv 파일 읽기 중 오류 발생: {e}")
            create_settings_file(settings_file)
            api_key = get_api_key_from_user()
            temperature = get_float_param_from_user(temperature_param_name, default_temperature, 0.0, 1.0)
            top_p = get_float_param_from_user(top_p_param_name, default_top_p, 0.0, 1.0)
            top_k = get_int_param_from_user(top_k_param_name, default_top_k, 1) # 최소값 1
            text_block_size = get_int_param_from_user(text_block_size_param_name, default_text_block_size, 1, 80000)
            previous_context_number = get_int_param_from_user(previous_context_number_param_name, default_previous_context_number, 0)
            retranslate_max_retries = get_int_param_from_user(retranslate_max_retries_param_name, default_retranslate_max_retries, 0, 5)
            second_translation = get_int_param_from_user(second_translation_param_name, default_second_translation, 1, 2)
            num_parallel = get_int_param_from_user(number_of_parallel_processing_param_name, default_num_parallel, 1)
            cover_image_modify = get_int_param_from_user(cover_image_modify_param_name, default_cover_image_modify, 1, 2)
            cover_text_position = get_int_param_from_user(cover_text_position_param_name, default_cover_text_position, 1, 4)
            cover_text = input(f"{cover_text_param_name} 값 입력 (기본값: {default_cover_text}): ") or default_cover_text
            font = input(f"{font_param_name} 값 입력 (기본값: {default_font}): ") or default_font
            font_size = get_int_param_from_user(font_size_param_name, default_font_size, 1)
            font_color = input(f"{font_color_param_name} 값 입력 (기본값: {default_font_color}): ") or default_font_color
            background_color = input(f"{background_color_param_name} 값 입력 (기본값: {default_background_color}): ") or default_background_color
            delete_temp_files = get_int_param_from_user(delete_temp_files_param_name, default_delete_temp_files, 1, 2)
            ridi_version = get_int_param_from_user(ridi_version_param_name, default_ridi_version, 1, 2)

            save_params_to_file(settings_file, api_key, temperature, top_p, top_k, text_block_size, previous_context_number, retranslate_max_retries, second_translation, num_parallel, cover_image_modify, cover_text_position, cover_text, font, font_size, font_color, background_color, delete_temp_files, ridi_version) 

    # 반환 값 튜플에 num_parallel 추가
    return (api_key, temperature, top_p, top_k, text_block_size, previous_context_number,
            retranslate_max_retries, second_translation, num_parallel,
            cover_image_modify, cover_text_position, cover_text, font,
            font_size, font_color, background_color, delete_temp_files,
            ridi_version)


def select_gemini_model(api_key):

    client = genai.Client(api_key=api_key)
    available_models = []
    available_models_display = []

    for m in client.models.list():
        if 'generateContent' in (getattr(m, 'supported_actions', None) or getattr(m, 'supported_generation_methods', [])):
            available_models.append(m.name)
            available_models_display.append(m.name.replace("models/", ""))

    print("\n사용 가능한 Gemini 모델:")
    for i, model_name in enumerate(available_models_display):
        print(f"{i+1}. {model_name}")

    while True:
        try:
            choice = input("\n사용할 모델 번호를 입력하세요 (Enter시 gemini-1.5-pro-002 사용): ")
            if not choice:
                default_model = "models/gemini-1.5-pro-002"
                for model_name in available_models:
                    if "gemini-1.5-pro-002" in model_name:
                        default_model = model_name
                        break
                print(f"gemini-1.5-pro-002({default_model.replace('models/', '')})를 사용합니다.")
                return default_model

            choice_index = int(choice) - 1
            if 0 <= choice_index < len(available_models):
                return available_models[choice_index]
            else:
                print("잘못된 번호입니다. 다시 입력해주세요.")
        except ValueError:
            print("숫자를 입력해야 합니다. 다시 입력해주세요.")



def load_prompt(filename="prompt.txt"):
    """프롬프트 파일을 읽어 내용을 반환합니다. 파일이 없으면 빈 문자열을 반환합니다."""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except FileNotFoundError:
        print(f"prompt.txt 파일을 찾을 수 없습니다. 추가 지침 없이 번역을 진행합니다.")
        return ""
    except Exception as e:
        print_colored(f"Error: 프롬프트 파일({filename})을 읽는 중 오류 발생: {e}", colorama.Fore.RED, colorama.Style.BRIGHT)
        return ""


def load_glossary(default_glossary_file="glossary.txt"):
    """
    사용자에게 용어집 파일 경로를 입력받거나, 기본 용어집 파일을 로드합니다.
    유효한 용어집 내용을 반환하거나, 없는 경우 빈 문자열을 반환합니다.
    경로/파일명에 공백이나 특수문자가 포함될 수 있도록 처리합니다.
    """
    glossary_content = ""

    while True:
        glossary_path = input("사용할 용어집 파일 경로를 입력하세요 (Enter시 기본값 사용): ").strip()
        if not glossary_path:
            glossary_path = default_glossary_file
            print_colored(f"기본 용어집 파일({glossary_path})을 사용합니다.", colorama.Fore.WHITE, colorama.Style.BRIGHT)
            break

        glossary_path = os.path.expandvars(glossary_path.strip().strip('"'))
        glossary_path = os.path.normpath(glossary_path)

        if os.path.exists(glossary_path):
            break
        else:
            print(f"용어집 파일({glossary_path})을 찾을 수 없습니다. 다시 입력해주세요.")

    if os.path.exists(glossary_path):
        try:
            with open(glossary_path, 'r', encoding='utf-8') as f:
                glossary_content = f.read().strip()
                if glossary_content:
                    print_colored("용어집이 로드되었습니다.", colorama.Fore.WHITE, colorama.Style.BRIGHT)
                    return glossary_content
                else:
                    print_colored("용어집 파일이 비어있습니다. 용어집 없이 번역합니다.", colorama.Fore.YELLOW, colorama.Style.BRIGHT)
                    return ""
        except Exception as e:
            print_colored(f"용어집 파일({glossary_path})을 읽는 중 오류 발생: {e}. 용어집 없이 번역합니다.", colorama.Fore.RED, colorama.Style.BRIGHT)
            return ""
    else:  # 파일이 없는 경우
        print_colored(f"용어집 파일({glossary_path})을 찾을 수 없습니다. 용어집 없이 번역합니다.", colorama.Fore.YELLOW, colorama.Style.BRIGHT)
        return ""


def load_character_dictionary():
    """
    사용자에게 캐릭터 사전 파일 경로를 입력받아 로드합니다.
    유효한 캐릭터 사전 내용을 반환하거나, 없는 경우 빈 딕셔너리를 반환합니다.
    """
    character_dictionary_content = {}

    while True:
        character_dictionary_path = input("사용할 캐릭터 사전 파일 경로를 입력하세요 (Enter시 미사용): ").strip()
        if not character_dictionary_path:
            print_colored("캐릭터 사전을 사용하지 않습니다.", colorama.Fore.YELLOW, colorama.Style.BRIGHT)
            return {}

        character_dictionary_path = os.path.expandvars(character_dictionary_path.strip().strip('"'))
        character_dictionary_path = os.path.normpath(character_dictionary_path)

        if os.path.exists(character_dictionary_path):
            break
        else:
            print(f"캐릭터 사전 파일({character_dictionary_path})을 찾을 수 없습니다. 다시 입력해주세요.")

    if os.path.exists(character_dictionary_path):
        try:
            with open(character_dictionary_path, 'r', encoding='utf-8') as f:
                character_dictionary_content = json.load(f)
                if character_dictionary_content:
                    print("캐릭터 사전이 로드되었습니다.")
                    return character_dictionary_content
                else:
                    print_colored("캐릭터 사전 파일이 비어있습니다. 캐릭터 사전 없이 번역합니다.", colorama.Fore.YELLOW, colorama.Style.BRIGHT)
                    return {}
        except Exception as e:
            print_colored(f"캐릭터 사전 파일({character_dictionary_path})을 읽는 중 오류 발생: {e}. 캐릭터 사전 없이 번역합니다.", colorama.Fore.RED, colorama.Style.BRIGHT)
            return {}
    else:  # 파일이 없는 경우
        print_colored(f"캐릭터 사전 파일({character_dictionary_path})을 찾을 수 없습니다. 캐릭터 사전 없이 번역합니다.", colorama.Fore.YELLOW, colorama.Style.BRIGHT)
        return {}
     

default_korean_style = """
p.korean_style {
    /* ... (기존 korean_style 내용) ... */
    color: #000000;
    font-family: "RIDIBatang";
    src: url('../Fonts/RIDIBatang.otf'); /* 경로는 EPUB 구조에 따라 달라질 수 있음 */
    font-size: 1.0em;
    font-style: normal;
    font-variant: normal;
    font-weight: normal;
    line-height: 1.8;
    margin-bottom: 0;
    margin-left: 0;
    margin-right: 0;
    margin-top: 0;
    orphans: 1;
    page-break-after: auto;
    page-break-before: auto;
    text-align: justify;
    text-decoration: none;
    text-indent: 1.0em; /* 들여쓰기 */
    text-transform: none;
    widows: 1;
}
"""

def load_korean_style(default_style):
    """korean_style.txt 파일을 읽거나 기본 스타일을 반환합니다."""
    try:
        with open("korean_style.txt", "r", encoding="utf-8") as f:
            style = f.read()
            #print("korean_style.txt 로드 성공.")
            logging.info("Loaded style from korean_style.txt")
            return style
    except FileNotFoundError:
        print_colored("Info: korean_style.txt 파일을 찾을 수 없어 기본 스타일을 사용합니다.", colorama.Fore.YELLOW)
        logging.info("korean_style.txt not found, using default style.")
        return default_style
    except Exception as e:
        print_colored(f"Warning: korean_style.txt 읽기 오류: {e}. 기본 스타일 사용.", colorama.Fore.YELLOW)
        logging.warning(f"Error reading korean_style.txt: {e}. Using default style.", exc_info=True)
        return default_style
        
        
def create_prompt(base_prompt_instructions, base_prompt_text, additional_instructions, glossary_content, character_dictionary, context):
    """
    번역 프롬프트를 생성합니다.
    """
    
    context["current_text"] = context["current_text"].replace('\n', '\n\n') # 줄바꿈 두 번으로 변경
    context["prev_context"] = context["prev_context"].replace('\n', '\n\n') # prev_context 줄바꿈 두 번
    
    prompt = base_prompt_instructions
    
    if additional_instructions:
        prompt += "\n\n" + additional_instructions
    if glossary_content:
        prompt += "\n\n[용어집]\n" + glossary_content + "\n\n번역 시, 용어집에 존재하는 단어는 항상 용어집을 사용해서 번역해주세요. 쉼표 앞의 단어는 원단어, 쉼표 뒤의 단어는 원단어의 번역어입니다. 용어집의 원단어와 일치하지 않는 단어는 용어집을 사용하면 안됩니다."
    if character_dictionary:
        character_info_prompt = ""
        character_info_prompt += "\n\n[캐릭터 사전]\n"
        if "characters" in character_dictionary:
            character_info_prompt += "\n[캐릭터 정보]\n"
            for char in character_dictionary["characters"]:
                character_info_prompt += f"- 이름 (원문): {char.get('name_original', '')}\n"
                character_info_prompt += f"  이름 (번역): {char.get('name_translated', '')}\n"
                character_info_prompt += f"  성별: {char.get('gender', '')}\n"
                character_info_prompt += f"  설명: {char.get('description', '')}\n"
                if char.get("nicknames"):
                    character_info_prompt += "  별칭:\n"
                    for nickname in char["nicknames"]:
                        character_info_prompt += f"    - {nickname.get('original', '')} ({nickname.get('translated', '')})\n"
                if char.get("relationships"):
                    character_info_prompt += "  관계:\n"
                    for rel in char["relationships"]:
                        target_id = rel.get('target', '')
                        target_name = ""
                        # relationships의 target에 대응하는 name_original을 characters에서 찾음
                        for c in character_dictionary["characters"]:
                            if c.get("id") == target_id:
                                target_name = c.get("name_original", "")
                                break
                        character_info_prompt += f"    - {target_name}: {rel.get('relation', '')}\n"
        if "groups" in character_dictionary:
            character_info_prompt += "\n[그룹 정보]\n"
            for group in character_dictionary["groups"]:
                character_info_prompt += f"- 그룹 ID: {group.get('group_id', '')}\n"
                character_info_prompt += f"  타입: {group.get('type', '')}\n"
                character_info_prompt += f"  이름 (원문): {group.get('name_original', '')}\n"
                character_info_prompt += f"  이름 (번역): {group.get('name_translated', '')}\n"
                if group.get("members"):
                    character_info_prompt += "  멤버:\n"
                    for member_id in group["members"]:
                        member_name = ""
                        # groups의 members에 대응하는 name_original을 characters에서 찾음
                        for c in character_dictionary["characters"]:
                            if c.get("id") == member_id:
                                member_name = c.get("name_original", "")
                                break
                        character_info_prompt += f"    - {member_name}\n"
        prompt += character_info_prompt + "\n\n번역 시, 캐릭터 사전에 존재하는 정보(이름, 별칭, 관계 등)를 최대한 활용해주세요. 특히 캐릭터 사전에 등장하는 인물(character)이나 단체(group)의 원이름(name_original, original in nicknames)은 항상 매칭되는 번역어(name_translated, translated in nicknames)로 번역해야 합니다. 호칭 번역 시에도 캐릭터 사전 정보를 최대한 활용하세요."

    prompt += "\n\n" + base_prompt_text.format(**context)  # 번역할 문장 (나중에 추가)
    
    return prompt


def create_prompt_2nd(additional_instructions, glossary_content, character_dictionary, context):
    """2차 번역용 프롬프트 생성 함수"""

    prompt = SECOND_TRANSLATION_PROMPT_INSTRUCTIONS
    
    if additional_instructions:
        prompt += "\n\n" + additional_instructions
    if glossary_content:
        prompt += "\n\n[용어집]\n" + glossary_content + "\n\n2차 번역 시, 용어집 단어를 참고해주세요. 쉼표 앞의 단어는 원단어, 쉼표 뒤의 단어는 원단어의 번역어입니다. 2차 번역 시, 1차 초벌 번역본에 존재하는 용어집에 있는 번역어는 수정하지 않습니다."
    if character_dictionary:
        character_info_prompt = ""
        character_info_prompt += "\n\n[캐릭터 사전]\n"
        if "characters" in character_dictionary:
            character_info_prompt += "\n[캐릭터 정보]\n"
            for char in character_dictionary["characters"]:
                character_info_prompt += f"- 이름 (원문): {char.get('name_original', '')}\n"
                character_info_prompt += f"  이름 (번역): {char.get('name_translated', '')}\n"
                character_info_prompt += f"  성별: {char.get('gender', '')}\n"
                character_info_prompt += f"  설명: {char.get('description', '')}\n"
                if char.get("nicknames"):
                    character_info_prompt += "  별칭:\n"
                    for nickname in char["nicknames"]:
                        character_info_prompt += f"    - {nickname.get('original', '')} ({nickname.get('translated', '')})\n"
                if char.get("relationships"):
                    character_info_prompt += "  관계:\n"
                    for rel in char["relationships"]:
                        target_id = rel.get('target', '')
                        target_name = ""
                        # relationships의 target에 대응하는 name_original을 characters에서 찾음
                        for c in character_dictionary["characters"]:
                            if c.get("id") == target_id:
                                target_name = c.get("name_original", "")
                                break
                        character_info_prompt += f"    - {target_name}: {rel.get('relation', '')}\n"
        if "groups" in character_dictionary:
            character_info_prompt += "\n[그룹 정보]\n"
            for group in character_dictionary["groups"]:
                character_info_prompt += f"- 그룹 ID: {group.get('group_id', '')}\n"
                character_info_prompt += f"  타입: {group.get('type', '')}\n"
                character_info_prompt += f"  이름 (원문): {group.get('name_original', '')}\n"
                character_info_prompt += f"  이름 (번역): {group.get('name_translated', '')}\n"
                if group.get("members"):
                    character_info_prompt += "  멤버:\n"
                    for member_id in group["members"]:
                        member_name = ""
                        # groups의 members에 대응하는 name_original을 characters에서 찾음
                        for c in character_dictionary["characters"]:
                            if c.get("id") == member_id:
                                member_name = c.get("name_original", "")
                                break
                        character_info_prompt += f"    - {member_name}\n"
        prompt += character_info_prompt + "\n\n2차 번역 시, 캐릭터 사전에 존재하는 정보를 참고하세요. 1차 초벌 번역본에는 이미 번역된 이름이 사용되고 있을 것이므로, 번역된 이름을 기준으로 캐릭터를 파악하고, 인물간의 관계를 파악합니다."
        
    prompt += "\n\n" + SECOND_TRANSLATION_PROMPT_TEXT.format(**context)
    
    return prompt
    
    
def detect_japanese_or_chinese(text):
    """텍스트에 일본어 또는 한자가 포함되어 있는지 확인합니다."""
    exclude_chars = ["・", "ー"]  # 제외할 문자 리스트 (함수 내부에 정의)

    temp_text = text
    for char in exclude_chars:
        temp_text = temp_text.replace(char, "")

    japanese_regex = re.compile(r'[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]')
    return bool(temp_text.strip() and japanese_regex.search(temp_text))


# --- 2. EPUB 파일 처리 (저수준) ---

def find_opf_file(epub_file):
    for item in epub_file.infolist():
        if item.filename.endswith(".opf"):
            return item.filename
    return None


def get_xhtml_files_from_opf(epub_file, opf_path):
    try:
        with epub_file.open(opf_path, 'r') as opf_content_file:
            opf_content = opf_content_file.read()
            soup = BeautifulSoup(opf_content, 'xml')
            manifest = soup.find('manifest')
            if not manifest:
                raise ValueError("OPF 파일에서 <manifest> 태그를 찾을 수 없습니다.")

            items = manifest.find_all('item', attrs={'media-type': 'application/xhtml+xml'}) or \
                    manifest.find_all('item', attrs={'media-type': 'application/x-dtbook+xml'})

            if not items:
                raise ValueError("OPF 파일에서 xhtml item을 찾을 수 없습니다.")

            spine = soup.find('spine')
            if spine and spine.get('toc'):
                itemrefs = spine.find_all('itemref')
                if itemrefs:
                    item_ids = [itemref.get('idref') for itemref in itemrefs]
                    item_map = {item.get('id'): item.get('href') for item in items}
                    xhtml_files = [os.path.basename(item_map.get(item_id)) for item_id in item_ids if
                                   item_map.get(item_id)]
                    return xhtml_files
                else:
                    xhtml_files = [os.path.basename(item.get('href')) for item in items]
                    return xhtml_files
            else:
                xhtml_files = [os.path.basename(item.get('href')) for item in items]
                return xhtml_files

    except Exception as e:
        print_colored(f"Error: OPF 파일 처리 중 오류 발생: {e}", colorama.Fore.RED, colorama.Style.BRIGHT)
        return None


# --- 3. xhtml 파일 파싱 및 내용 추출 ---

def parse_xhtml_file(epub_file, xhtml_file_path, text_block_counter, output_dir, text_block_size):
    """
    XHTML 파일을 파싱하여 내용을 추출하고 텍스트 블록과 이미지 블록으로 나눕니다.
    <p><img></p> 와 같은 구조는 <p>를 이미지 블록으로 처리합니다.
    """
    try:
        # --- 파일 경로 및 네비게이션 파일 확인 (이전과 동일) ---
        xhtml_filename_only = os.path.basename(xhtml_file_path)
        common_nav_filenames = {"navigation-documents.xhtml", "nav.xhtml", "toc.xhtml"}
        if xhtml_filename_only.lower() in common_nav_filenames:
            logging.info(f"Skipping navigation file: {xhtml_file_path}")
            return None, text_block_counter

        # --- EPUB 내에서 파일 찾고 읽기 (인코딩 처리 포함, 이전과 동일) ---
        xhtml_data = None; found_item = None
        target_path_normalized = os.path.normpath(xhtml_file_path).replace('\\', '/')
        opf_path = find_opf_file(epub_file); base_dir = os.path.dirname(opf_path) if opf_path else ''
        potential_path_in_zip = os.path.normpath(os.path.join(base_dir, xhtml_file_path)).replace('\\', '/')

        for item in epub_file.infolist():
            item_path_normalized = os.path.normpath(item.filename).replace('\\', '/')
            if item_path_normalized == potential_path_in_zip or item_path_normalized == target_path_normalized: found_item = item; break
            elif os.path.basename(item_path_normalized).lower() == xhtml_filename_only.lower(): found_item = item; break

        if found_item:
             with epub_file.open(found_item) as f: xhtml_data = f.read()
             detected_encoding = 'utf-8'; xhtml_string = None
             if xhtml_data.startswith(b'\xef\xbb\xbf'): xhtml_data_no_bom = xhtml_data[3:]
             else: xhtml_data_no_bom = xhtml_data
             try: xhtml_string = xhtml_data_no_bom.decode(detected_encoding)
             except UnicodeDecodeError:
                 try: detected_encoding = 'cp949'; xhtml_string = xhtml_data_no_bom.decode(detected_encoding); logging.info(f"Decoded {found_item.filename} using {detected_encoding}")
                 except Exception as decode_err: logging.error(f"Failed to decode {found_item.filename}: {decode_err}"); return None, text_block_counter
             except Exception as generic_decode_err: logging.error(f"Error decoding {found_item.filename}: {generic_decode_err}"); return None, text_block_counter
             if xhtml_string is None: logging.error(f"Decoding failed for {found_item.filename}"); return None, text_block_counter
        else: logging.warning(f"Could not find XHTML file {xhtml_file_path} in EPUB. Skipping."); return None, text_block_counter
        # --- 파일 찾기 및 읽기 끝 ---

        soup = BeautifulSoup(xhtml_string, 'html.parser')
        body = soup.find('body')
        if not body or not body.get_text(strip=True):
            logging.info(f"Skipping XHTML file {xhtml_file_path} due to missing or empty body.")
            return None, text_block_counter

        xhtml_filename = os.path.basename(found_item.filename)
        xhtml_json = {}
        xhtml_json_list = []

        # --- before_body, end_body 추출 및 저장 (이전과 동일) ---
        before_body_content = xhtml_string.split('<body', 1)[0]; body_tag_match = re.search(r'<body[^>]*>', xhtml_string); body_tag = body_tag_match.group(0) if body_tag_match else '<body>'
        before_body_content += body_tag; before_body_filename = os.path.join(output_dir, f"{os.path.splitext(xhtml_filename)[0]}_before_body.txt")
        save_txt_file(before_body_content, before_body_filename); xhtml_json_list.append({"type": "before_body", "content": before_body_filename})

        end_body_split = xhtml_string.split('</body>', 1)
        if len(end_body_split) > 1: end_body_content = "</body>" + end_body_split[1]
        else: html_split = xhtml_string.split('</html>', 1); end_body_content = "</html>" + html_split[1] if len(html_split) > 1 else ""
        end_body_filename = os.path.join(output_dir, f"{os.path.splitext(xhtml_filename)[0]}_end_body.txt")
        save_txt_file(end_body_content, end_body_filename); # end_body는 json_list에 나중에 추가
        # --- before_body, end_body 끝 ---

        text_block_content = ""
        image_count = 1
        text_block_count_in_xhtml = 1
        current_char_count = 0

        # --- 노드 처리 헬퍼 함수 ---
        def process_node(node, current_char_count, text_block_content, xhtml_json_list, text_block_counter, text_block_count_in_xhtml, image_count, current_xhtml_filename):

            # 1. 텍스트 노드 처리
            if isinstance(node, NavigableString):
                stripped_text = node.strip()
                if stripped_text:
                    # 현재 텍스트 블록에 추가 (사이즈 확인 후)
                    element_html = f"<p>{stripped_text}</p>" # 일관성을 위해 p로 감싸기 (선택적)
                    element_char_count = len(stripped_text)
                    if current_char_count + element_char_count > text_block_size and current_char_count > 0:
                        xhtml_json_list, text_block_counter = add_text_block(xhtml_json_list, text_block_content, text_block_counter, text_block_count_in_xhtml, output_dir, current_xhtml_filename)
                        text_block_count_in_xhtml += 1
                        text_block_content = element_html + "\n"
                        current_char_count = element_char_count
                    else:
                        text_block_content += element_html + "\n"
                        current_char_count += element_char_count
                # 텍스트 노드는 항상 처리하고 상태 반환
                return current_char_count, text_block_content, xhtml_json_list, text_block_counter, text_block_count_in_xhtml, image_count

            # --- 요소 노드 처리 ---
            if not hasattr(node, 'name'): # 이름 없는 노드(주석 등) 건너뛰기
                return current_char_count, text_block_content, xhtml_json_list, text_block_counter, text_block_count_in_xhtml, image_count

            is_image_block = False
            image_block_element = None

            # 2. 현재 노드가 <p> 또는 <div> 이면서 순수 이미지 컨테이너인지 확인
            if node.name in ['p', 'div']:
                img_children = node.find_all('img', recursive=False)
                if len(img_children) == 1: # 직계 자식으로 img가 하나
                    # 내부 내용이 img와 공백 문자로만 이루어져 있는지 확인
                    only_img_and_whitespace = True
                    for child in node.contents:
                        if child == img_children[0]: continue # img 자체는 건너뜀
                        if isinstance(child, NavigableString) and not child.strip(): continue # 공백 문자 건너뜀
                        # 다른 태그나 텍스트가 있으면 순수 이미지 컨테이너 아님
                        only_img_and_whitespace = False; break
                    if only_img_and_whitespace:
                        is_image_block = True
                        image_block_element = node # 이 <p> 또는 <div>가 이미지 블록
                        logging.debug(f"Identified image block: <{node.name}> (contains only img) in {current_xhtml_filename}")


            # 3. 현재 노드가 <img> 이고, (2번 경우 제외) 독립적인 이미지 블록인지 확인
            elif node.name == 'img':
                # 이 img가 이미 처리된 p/div 안에 있는지 확인 (간접적 확인)
                # -> 만약 부모 p/div가 순수 이미지 컨테이너였다면, 그 부모 처리 시 is_image_block=True 가 되어 아래 로직 실행 안됨
                # -> 따라서 이 로직은 부모가 순수 이미지 컨테이너가 *아닌* 경우의 img에 해당
                is_image_block = True
                image_block_element = node # img 태그 자체를 블록으로
                logging.debug(f"Identified image block: <{node.name}> (standalone or inside complex parent) in {current_xhtml_filename}")


            # --- 이미지 블록 처리 ---
            if is_image_block:
                # 기존 텍스트 블록 저장
                if text_block_content and clean_text_from_tags(text_block_content).strip():
                    logging.debug(f"Saving preceding text block before image block {image_count}")
                    xhtml_json_list, text_block_counter = add_text_block(
                        xhtml_json_list, text_block_content, text_block_counter,
                        text_block_count_in_xhtml, output_dir, current_xhtml_filename
                    )
                    text_block_count_in_xhtml += 1
                # 텍스트 관련 변수 초기화
                text_block_content = ""
                current_char_count = 0
                # 이미지 블록 추가
                xhtml_json_list = add_image_block(
                    xhtml_json_list, image_block_element, current_xhtml_filename, image_count, output_dir
                )
                image_count += 1
                # 이미지 블록 처리 후 상태 반환
                return current_char_count, text_block_content, xhtml_json_list, text_block_counter, text_block_count_in_xhtml, image_count


            # --- 이미지 블록이 아닌 요소 처리 ---

            # 4. 텍스트 컨테이너 (<p>, <h1>-<h6>) 처리
            elif node.name in ['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                 element_html = str(node)
                 element_text_check = clean_text_from_tags(element_html)
                 if not element_text_check.strip(): # 내용 없는 태그 건너뛰기
                      logging.debug(f"Skipping empty element: <{node.name}> in {current_xhtml_filename}")
                 else:
                     element_char_count = len(element_text_check)
                     # 텍스트 블록 크기 확인 및 추가/분할
                     if current_char_count + element_char_count > text_block_size and current_char_count > 0:
                         logging.debug(f"Text block limit reached. Saving block {text_block_counter}")
                         xhtml_json_list, text_block_counter = add_text_block(xhtml_json_list, text_block_content, text_block_counter, text_block_count_in_xhtml, output_dir, current_xhtml_filename)
                         text_block_count_in_xhtml += 1
                         text_block_content = element_html + "\n"
                         current_char_count = element_char_count
                     else:
                         text_block_content += element_html + "\n"
                         current_char_count += element_char_count
                 # 텍스트 컨테이너 처리 후 상태 반환
                 return current_char_count, text_block_content, xhtml_json_list, text_block_counter, text_block_count_in_xhtml, image_count

            # 5. <div> (이미지 컨테이너 아닌 경우) 및 기타 컨테이너 태그 처리 -> 자식 노드 재귀 호출
            elif hasattr(node, 'contents'): # 자식이 있는 다른 태그들 (div 포함)
                 logging.debug(f"Traversing children of container tag: <{node.name}> in {current_xhtml_filename}")
                 for child in node.contents:
                      # 재귀 호출로 상태 업데이트
                      current_char_count, text_block_content, xhtml_json_list, text_block_counter, text_block_count_in_xhtml, image_count = process_node(
                          child, current_char_count, text_block_content, xhtml_json_list, text_block_counter, text_block_count_in_xhtml, image_count, current_xhtml_filename
                      )
                 # 자식 처리 후 누적된 상태 반환
                 return current_char_count, text_block_content, xhtml_json_list, text_block_counter, text_block_count_in_xhtml, image_count

            # 6. 처리되지 않은 노드 (자식도 없는 단일 태그 등) - 상태 유지하고 반환
            logging.debug(f"Node <{node.name}> not explicitly handled, returning current state.")
            return current_char_count, text_block_content, xhtml_json_list, text_block_counter, text_block_count_in_xhtml, image_count
        # --- process_node 함수 끝 ---

        # --- body 자식 노드 순회 시작 ---
        if body:
            logging.info(f"Starting node processing for {xhtml_filename}")
            for node in body.contents:
                 current_char_count, text_block_content, xhtml_json_list, text_block_counter, text_block_count_in_xhtml, image_count = process_node(
                    node, current_char_count, text_block_content, xhtml_json_list, text_block_counter, text_block_count_in_xhtml, image_count, xhtml_filename
                 )
            logging.info(f"Finished node processing for {xhtml_filename}")
        # --- body 자식 노드 순회 끝 ---

        # 남은 텍스트 블록 저장
        if text_block_content and clean_text_from_tags(text_block_content).strip():
            logging.debug(f"Saving final remaining text block {text_block_counter}")
            xhtml_json_list, text_block_counter = add_text_block(
                xhtml_json_list, text_block_content, text_block_counter,
                text_block_count_in_xhtml, output_dir, xhtml_filename
            )

        # end_body 정보 추가 (파일은 위에서 저장됨)
        xhtml_json_list.append({
            "type": "end_body",
            "content": end_body_filename
        })

        xhtml_json[xhtml_filename] = xhtml_json_list
        logging.info(f"Successfully parsed and processed XHTML file: {xhtml_filename}")
        return xhtml_json, text_block_counter

    # --- 예외 처리 (이전과 동일) ---
    except FileNotFoundError as e: logging.warning(f"FileNotFoundError handled for {xhtml_file_path}: {e}"); return None, text_block_counter
    except Exception as e: import traceback; logging.error(f"Exception processing XHTML file '{xhtml_file_path}': {e}\n{traceback.format_exc()}", exc_info=False); return None, text_block_counter



BLOCK_START_PH = "___BLOCK_START___"
BLOCK_END_PH = "___BLOCK_END___"
BR_PH = "___BR___"


    
    


def add_text_block(xhtml_json_list, text_block_content, text_block_counter,
                   text_block_count_in_xhtml, output_dir, xhtml_filename):
    """텍스트 블록을 저장하고 JSON 목록에 추가합니다."""
    # 파일 이름 생성
    text_block_filename = os.path.join(output_dir, f"text_block_{text_block_counter}.txt")

    # 태그 제거 및 텍스트 정제 (빈 줄 제거 포함)
    cleaned_text_content = clean_text_from_tags(text_block_content)
    lines = cleaned_text_content.splitlines()
    non_empty_lines = [line for line in lines if line.strip()]
    final_text_content = "\n".join(non_empty_lines)

    # 내용이 있을 때만 파일 저장 및 JSON 추가
    if final_text_content:
        save_txt_file(final_text_content, text_block_filename)

        # 원본 파일 복사 (존재하지 않으면 생성)
        origin_filename = os.path.join(output_dir, f"text_block_{text_block_counter}.origin.txt")
        if not os.path.exists(origin_filename):
             # 원본 저장을 위해 text_block_content (HTML 포함) 사용 고려 가능
             # 여기서는 정제된 텍스트를 원본으로 저장 (기존 로직 유지)
             save_txt_file(final_text_content, origin_filename)
             # shutil.copy2(text_block_filename, origin_filename) # 정제된 파일 복사

        xhtml_json_list.append({
            "type": "text_block",
            "content": text_block_filename
        })
        logging.debug(f"Added text block: {text_block_filename}")
        return xhtml_json_list, text_block_counter + 1
    else:
        logging.debug(f"Skipped saving empty text block derived from: {text_block_content[:100]}...")
        # 카운터는 증가시키지 않음
        return xhtml_json_list, text_block_counter


def add_image_block(xhtml_json_list, element, xhtml_filename, image_count, output_dir):
    """이미지 블록 요소의 HTML 표현을 저장합니다."""
    image_block_filename = os.path.join(output_dir, f"{os.path.splitext(xhtml_filename)[0]}_image_{image_count}.txt")
    image_content = str(element) # 전달된 요소(img, p, div)의 HTML 저장
    save_txt_file(image_content, image_block_filename)
    xhtml_json_list.append({
        "type": "image",
        "content": image_block_filename
    })
    logging.debug(f"Added image block: {image_block_filename} for element <{element.name}>")
    return xhtml_json_list


def save_txt_file(content, filename):
    try:
        with open(filename, 'w', encoding='utf-8') as txt_file:
            txt_file.write(content)
    except Exception as e:
        print_colored(f"Error: TXT 파일 저장 오류: {e}", colorama.Fore.RED, colorama.Style.BRIGHT)
        logging.error(f"Error saving TXT file {filename}: {e}", exc_info=True)


def save_json_file(data, filename):
    try:
        with open(filename, 'w', encoding='utf-8') as json_file:
            json.dump(data, json_file, indent=4, ensure_ascii=False)
    except Exception as e:
        print_colored(f"Error: JSON 파일 저장 오류: {e}", colorama.Fore.RED, colorama.Style.BRIGHT)


def extract_epub_content(epub_path, selected_model, api_key, temperature, top_p, top_k, glossary_content, cover_image_modify, cover_text_position, cover_text, font_path, font_size, font_color, background_color, retranslate_max_retries, previous_context_number, text_block_size, num_parallel, character_dictionary={}):
    output_dir = None
    try:
        epub_filename = os.path.splitext(os.path.basename(epub_path))[0]
        script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()

        base_output_dir = os.path.join(script_dir, "temp_translations", epub_filename)
        counter = 1
        output_dir = base_output_dir
        while os.path.exists(output_dir):
            output_dir = f"{base_output_dir} ({counter})"
            counter += 1
        os.makedirs(output_dir, exist_ok=True)

        log_filepath = os.path.join(output_dir, "log.txt")
        root_logger = logging.getLogger()
        if root_logger.hasHandlers():
            for handler in root_logger.handlers[:]: root_logger.removeHandler(handler)
        logging.basicConfig(filename=log_filepath, level=logging.INFO, encoding='utf-8', format='%(asctime)s - %(levelname)s - %(message)s')
        logging.info(f"Starting EPUB processing for: {epub_path}")
        logging.info(f"Output directory: {output_dir}")

        with zipfile.ZipFile(epub_path, 'r') as epub_file:
            opf_path = find_opf_file(epub_file)
            if not opf_path: raise FileNotFoundError("OPF 파일을 찾을 수 없습니다.")
            logging.info(f"Found OPF file: {opf_path}")

            xhtml_files = get_xhtml_files_from_opf(epub_file, opf_path)
            if not xhtml_files: raise FileNotFoundError("OPF 파일에서 xhtml 파일 목록을 추출할 수 없습니다.")
            logging.info(f"Found {len(xhtml_files)} XHTML files in OPF.")

            json_data = {"epub_filename": epub_filename}
            text_block_counter = 1
            logging.info("Parsing XHTML files...")
            for xhtml_file_path_in_opf in xhtml_files:
                xhtml_json, text_block_counter = parse_xhtml_file(epub_file, xhtml_file_path_in_opf, text_block_counter, output_dir, text_block_size)
                if xhtml_json:
                    actual_filename_key = list(xhtml_json.keys())[0]
                    json_data[actual_filename_key] = xhtml_json[actual_filename_key]

            json_filename = os.path.join(output_dir, epub_filename + ".json")
            save_json_file(json_data, json_filename)
            logging.info(f"Saved structure JSON to {json_filename}")

            create_line_level_json(output_dir, json_data)
            logging.info("Created line level JSON (all_lines.json)")

            total_input_chars = 0
            total_output_chars = 0
            failed_files = []

            logging.info("Starting text block translation...")
            # 1차 번역 실행 (settings.csv의 num_parallel 사용)
            failed_files, total_input_chars, total_output_chars = translate_text_blocks(
                output_dir, json_data, selected_model, api_key, temperature, top_p, top_k,
                base_prompt_instructions,
                base_prompt_text,
                glossary_content=glossary_content,
                character_dictionary=character_dictionary,
                previous_context_number=previous_context_number,
                num_parallel=num_parallel # settings.csv 값 전달
                # retry_count, retry_delay, request_delay use defaults
            )
            logging.info(f"Initial block translation finished. Failures: {len(failed_files)}. Input chars: {total_input_chars}. Output chars: {total_output_chars}.")

            estimated_retranslate_cost = 0

            temp_json_path = os.path.join(output_dir, "temp_translation.json")

            if not os.path.exists(temp_json_path):
                print_colored(f"Error: {temp_json_path} 파일을 찾을 수 없습니다. 줄 수 비교 재번역을 건너뜁니다.", colorama.Fore.RED, colorama.Style.BRIGHT)
                input_chars_added_lc, output_chars_added_lc, retranslate_cost_lc = 0, 0, 0
            else:
                try:
                    with open(temp_json_path, 'r', encoding='utf-8') as f:
                        translation_data = json.load(f)

                    # QC 1: 줄 수 비교 재번역 (settings.csv의 num_parallel 사용)
                    input_chars_added_lc, output_chars_added_lc, retranslate_cost_lc = retranslate_by_line_count(
                        output_dir, selected_model, api_key, temperature, top_p, top_k,
                        base_prompt_instructions, glossary_content, character_dictionary,
                        previous_context_number, retranslate_max_retries, translation_data,
                        num_parallel=num_parallel # settings.csv 값 전달
                    )
                    total_input_chars += input_chars_added_lc
                    total_output_chars += output_chars_added_lc
                    estimated_retranslate_cost += retranslate_cost_lc

                except Exception as e:
                    print_colored(f"Error: {temp_json_path} 로드 또는 처리 중 오류 발생: {e}. 줄 수 비교 재번역을 건너뜁니다.", colorama.Fore.RED, colorama.Style.BRIGHT)
                    input_chars_added_lc, output_chars_added_lc, retranslate_cost_lc = 0, 0, 0




            logging.info("EPUB content extraction and translation phases complete.")
            return json_data, output_dir, failed_files, total_input_chars, total_output_chars

    except zipfile.BadZipFile:
        print_colored(f"Error: {epub_path}는 유효한 EPUB 파일이 아닙니다.", colorama.Fore.RED, colorama.Style.BRIGHT)
        logging.error(f"BadZipFile error for {epub_path}")
        if output_dir and os.path.exists(output_dir): # 실패 시 생성된 output_dir이 있으면 반환
            return None, output_dir, [], 0, 0
        else:
            return None, None, [], 0, 0 # output_dir도 없으면 None 반환
    except FileNotFoundError as e:
        print_colored(f"Error: {e}", colorama.Fore.RED, colorama.Style.BRIGHT)
        logging.error(f"FileNotFoundError: {e}")
        if output_dir and os.path.exists(output_dir):
            return None, output_dir, [], 0, 0
        else:
            return None, None, [], 0, 0
    except Exception as e:
        print_colored(f"Error: EPUB 파일 처리 중 오류 발생: {e}", colorama.Fore.RED, colorama.Style.BRIGHT)
        logging.error(f"Exception during EPUB processing: {e}", exc_info=True)
        if output_dir and os.path.exists(output_dir):
            return None, output_dir, [], 0, 0
        else:
            return None, None, [], 0, 0
            
        
def create_line_level_json(output_dir, json_data):
    """
    모든 text_block_*.txt 파일을 읽어 줄 단위 정보를 담은 JSON 파일을 생성합니다.
    (빈 줄 제외, 파일별 리스트 구조)
    """
    all_lines = {}  # 딕셔너리로 변경
    text_block_files = []

    # text_block_*.txt 파일 목록 가져오기
    for filename in json_data:
        if filename == "epub_filename":
            continue
        for block in json_data[filename]:
            if block["type"] == "text_block":
                text_block_files.append(block["content"])

    # 파일들을 숫자 순서대로 정렬
    text_block_files.sort(key=lambda f: int(re.search(r'\d+', os.path.basename(f)).group()))

    # 모든 파일을 열어서 줄 단위로 읽고 정보를 저장 (빈 줄 제외, 파일별 리스트)
    for file_path in text_block_files:
        file_name = os.path.basename(file_path)
        all_lines[file_name] = [] # 빈 리스트 생성 (파일별)
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line_num, line_content in enumerate(lines):
                stripped_line = line_content.strip()
                if stripped_line:
                    all_lines[file_name].append({  # 해당 파일 리스트에 추가
                        "line_number": line_num + 1,
                        "content": stripped_line
                    })

    # JSON 파일로 저장
    line_level_json_path = os.path.join(output_dir, "all_lines.json")
    with open(line_level_json_path, 'w', encoding='utf-8') as f:
        json.dump(all_lines, f, indent=4, ensure_ascii=False)

    
# --- 4. 번역 (블록/줄 단위) ---
        
def create_translation_json(output_dir, json_data, previous_context_number):
    """
    번역을 위한 JSON 파일을 생성합니다. (이전 문맥 포함 및 리스트 인덱스 추가)
    """
    line_level_json_path = os.path.join(output_dir, "all_lines.json")
    # all_lines.json 로드 시 예외 처리 추가 가능
    try:
        with open(line_level_json_path, 'r', encoding='utf-8') as f:
            all_lines = json.load(f)
    except FileNotFoundError:
        logging.error(f"{line_level_json_path} not found. Cannot determine context.")
        all_lines = {} # 빈 딕셔너리로 진행 (문맥 없이)
    except json.JSONDecodeError as json_err:
         logging.error(f"Error decoding {line_level_json_path}: {json_err}. Cannot determine context.")
         all_lines = {}
    except Exception as read_err:
         logging.error(f"Error reading {line_level_json_path}: {read_err}. Cannot determine context.")
         all_lines = {}


    text_block_files = []
    # json_data 순회하며 text_block 경로 수집
    for filename_key in json_data:
        if filename_key == "epub_filename": continue
        if isinstance(json_data[filename_key], list):
            for block in json_data[filename_key]:
                if isinstance(block, dict) and block.get("type") == "text_block":
                    content_path = block.get("content")
                    if isinstance(content_path, str): # 경로가 문자열인지 확인
                        text_block_files.append(content_path) # 전체 경로 저장
                    else:
                        logging.warning(f"Invalid content path found in json_data for key {filename_key}: {content_path}")
        else:
             logging.warning(f"Unexpected data structure for key {filename_key} in json_data (expected list): {type(json_data[filename_key])}")


    # 파일들을 숫자 순서대로 정렬 (전체 경로에서 파일명 숫자 추출)
    def get_block_number(filepath):
        match = re.search(r'text_block_(\d+)\.txt$', os.path.basename(filepath))
        return int(match.group(1)) if match else float('inf')
    text_block_files.sort(key=get_block_number)


    translation_data = []
    total_files_count = len(text_block_files) # 실제 텍스트 블록 파일 수

    # 전체 텍스트 블록 파일 이름 목록 (문맥 구성용)
    text_block_basenames_ordered = [os.path.basename(f) for f in text_block_files]

    for i, current_file_path in enumerate(text_block_files):
        current_file_basename = os.path.basename(current_file_path)

        # --- 이전 문맥 구성 (개선) ---
        prev_context_lines_content = []
        remaining_lines_needed = previous_context_number
        current_index_in_basenames = text_block_basenames_ordered.index(current_file_basename) # 정렬된 리스트에서 현재 파일 인덱스 찾기

        idx_pointer = current_index_in_basenames - 1
        while remaining_lines_needed > 0 and idx_pointer >= 0:
            prev_file_basename = text_block_basenames_ordered[idx_pointer]
            if prev_file_basename in all_lines: # all_lines는 basename을 키로 사용
                prev_file_lines_data = all_lines[prev_file_basename] # 해당 파일의 라인 데이터 리스트 [{line_num, content}, ...]
                lines_to_take = min(remaining_lines_needed, len(prev_file_lines_data))
                # 리스트 앞에 추가 (올바른 순서 유지)
                prev_context_lines_content[0:0] = [line['content'] for line in prev_file_lines_data[-lines_to_take:]]
                remaining_lines_needed -= lines_to_take
            else:
                logging.warning(f"Context file {prev_file_basename} not found in all_lines.json")
            idx_pointer -= 1

        prev_context = "\n".join(prev_context_lines_content)
        # --- 이전 문맥 구성 끝 ---

        current_content = ""
        char_count = 0
        if os.path.exists(current_file_path):
          try:
              with open(current_file_path, 'r', encoding='utf-8') as f:
                  current_content = f.read()
                  char_count = len(current_content)
          except Exception as read_err:
               logging.error(f"Error reading current file {current_file_path}: {read_err}")
               # 오류 발생 시 기본값 유지 (current_content="", char_count=0)

        translation_data.append({
            "item_index_in_list": i, # ★★★ 리스트에서의 실제 인덱스 추가 ★★★
            "file_index": get_block_number(current_file_path), # 파일 이름의 번호 (기존 로직 유지)
            "filepath": current_file_path, # 전체 경로 저장
            "prev_context": prev_context,
            "current_text": current_content,
            "translated_text": "",
            "char_count": char_count,
            "status": "pending" # 초기 상태 추가 (선택적)
          })

    temp_json_path = os.path.join(output_dir, "temp_translation.json")
    try:
        with open(temp_json_path, 'w', encoding='utf-8') as f:
            json.dump(translation_data, f, indent=4, ensure_ascii=False)
    except Exception as write_err:
        logging.error(f"Error writing {temp_json_path}: {write_err}")
        # 파일 쓰기 실패 시 처리 (예: 예외 다시 발생)
        raise write_err

    return temp_json_path, total_files_count # 실제 파일 수 반환


def create_translation_json_for_lines(output_dir, line_files, original_filepath):
    """
    줄 단위 번역을 위한 translation JSON 파일을 생성합니다.
    """
    translation_data = []
    for i, line_filepath in enumerate(line_files):
        with open(line_filepath, 'r', encoding='utf-8') as f:
            current_text = f.read()

        prev_context = ""
        if i > 0:
            with open(line_files[i-1], 'r', encoding='utf-8') as f:
                prev_context = f.read()

        translation_data.append({
            "file_index": i + 1,
            "filepath": line_filepath,
            "prev_context": prev_context,
            "current_text": current_text,
            "translated_text": "",
            "char_count": len(current_text)
        })

    base_filename = os.path.splitext(os.path.basename(original_filepath))[0]
    line_translation_json_path = os.path.join(output_dir, f"{base_filename}_line_trans.json")  # 파일명 변경
    
    with open(line_translation_json_path, 'w', encoding='utf-8') as f:
        json.dump(translation_data, f, indent=4, ensure_ascii=False)

    return line_translation_json_path


def split_text_block_for_initial_translation(filepath):
    """
    텍스트 블록 파일을 줄 단위로 분할하여 개별 파일로 저장하고,
    순서 정보를 담은 JSON 파일을 생성합니다. (빈 줄 제외) - 첫 번역용
    """
    output_dir = os.path.dirname(filepath)
    base_filename = os.path.splitext(os.path.basename(filepath))[0]
    line_json_filepath = os.path.join(output_dir, f"{base_filename}.json")
    line_files = []
    line_data = []

    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    line_count = 0  # 실제 줄 번호 카운트 (빈 줄 제외)
    for i, line in enumerate(lines):
        line = line.strip(' \t\n\r')  # 공백, 탭, 개행 문자만 제거
        if not line:  # 빈 줄은 무시
            continue

        line_count += 1  # 빈 줄 아닐 때만 카운트 증가

        line_filename = os.path.join(output_dir, f"{base_filename}_{line_count}.txt")  # line_count 사용
        with open(line_filename, 'w', encoding='utf-8') as line_file:
            line_file.write(line)

        line_data.append({
            "order": line_count,  # line_count 사용
            "content": line
        })
        line_files.append(line_filename)

    with open(line_json_filepath, 'w', encoding='utf-8') as json_file:
        json.dump(line_data, json_file, indent=4, ensure_ascii=False)

    return line_json_filepath, line_files
    

def split_text_block_for_retranslation(filepath, retry_count):
    """
    텍스트 블록 파일을 줄 단위로 분할하여 개별 파일로 저장합니다. (빈 줄 제외) - 재번역용
    JSON 파일을 생성하지 않습니다.
    """
    output_dir = os.path.dirname(filepath)
    base_filename = os.path.splitext(os.path.basename(filepath))[0]
    line_files = []

    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    line_count = 0  # 실제 줄 번호 카운트 (빈 줄 제외)
    for i, line in enumerate(lines):
        line = line.strip(' \t\n\r')  # 공백, 탭, 개행 문자만 제거
        if not line:  # 빈 줄은 무시
            continue

        line_count += 1  # 빈 줄 아닐 때만 카운트 증가

        # 파일 이름 형식 변경
        line_filename = os.path.join(output_dir, f"{base_filename}_{line_count}.txt")
        with open(line_filename, 'w', encoding='utf-8') as line_file:
            line_file.write(line)

        line_files.append(line_filename)

    return line_files

    
def translate_text_blocks(output_dir, json_data, selected_model, api_key, temperature, top_p, top_k,
                          base_prompt_instructions, base_prompt_text,
                          retry_count=3, retry_delay=10, request_delay=0.6, glossary_content="",
                          character_dictionary={}, previous_context_number=5, num_parallel=3):
    temp_json_path, total_files = create_translation_json(output_dir, json_data, previous_context_number)

    generation_config = {"temperature": temperature, "top_p": top_p, "top_k": top_k}
    try:
        client = genai.Client(api_key=api_key)
        model = GeminiModel(client, model_name=selected_model, safety_settings=safety_settings, generation_config=generation_config)
    except Exception as model_err:
        print_colored(f"Error: Failed create Gemini model instance ({selected_model}): {model_err}", colorama.Fore.RED, colorama.Style.BRIGHT)
        return [], 0, 0

    print(f"\n번역 설정: temperature={temperature}, top_p={top_p}, top_k={top_k}, 병렬 번역 수={num_parallel}")

    translation_data = []
    try:
        with open(temp_json_path, 'r', encoding='utf-8') as f: translation_data = json.load(f)
        if not isinstance(translation_data, list): raise ValueError("Data not list.")
        invalid_items = []
        for i, item in enumerate(translation_data):
             if not isinstance(item, dict): invalid_items.append(f"Index {i} not dict"); continue
             filepath = item.get('filepath')
             current_text = item.get('current_text')
             if not isinstance(filepath, str): invalid_items.append(f"Index {i} invalid 'filepath'")
             if not isinstance(current_text, str): translation_data[i]['current_text'] = ''
             # Recalculate char_count based on current_text for consistency
             char_count_recalc = len(translation_data[i]['current_text'])
             if not isinstance(item.get('char_count'), int) or item.get('char_count') != char_count_recalc:
                 translation_data[i]['char_count'] = char_count_recalc

        if invalid_items:
            logging.error(f"Invalid structure {temp_json_path}:\n" + "\n".join(invalid_items))
            print_colored(f"Error: {temp_json_path} structure errors. Check logs.", colorama.Fore.RED); return [], 0, 0
    except FileNotFoundError: print_colored(f"Error: {temp_json_path} not found.", colorama.Fore.RED); return [], 0, 0
    except json.JSONDecodeError as json_err: print_colored(f"Error: {temp_json_path} JSON parse: {json_err}", colorama.Fore.RED); return [], 0, 0
    except Exception as load_err: print_colored(f"Error: Load exception {temp_json_path}: {load_err}", colorama.Fore.RED); return [], 0, 0


    total_non_whitespace_input_block = sum(count_non_whitespace(item.get("current_text", "")) for item in translation_data)
    if total_files == 0 or total_non_whitespace_input_block == 0:
        print_colored("변역할 텍스트가 없습니다.", colorama.Fore.YELLOW); return [], 0, 0

    print(f"총 텍스트 블록 수: {total_files}, 총 입력 글자 수: {total_non_whitespace_input_block}")

    start_time = time.time()
    total_input_chars = 0
    total_output_chars = 0
    failed_block_files = []
    line_translation_candidates = []
    translated_file_count_ref = [0]

    additional_instructions = load_prompt()
    max_workers = num_parallel

    with tqdm(total=total_non_whitespace_input_block, desc="번역 진행률 ", unit=" 자", dynamic_ncols=True, position=0, leave=True) as block_pbar:
        tqdm.write(f"\n[번역 1 단계] 텍스트 블록 번역 시작")
        futures_map = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            for i, item in enumerate(translation_data):
                future = executor.submit(
                    translate_single_block,
                    item, model, base_prompt_instructions, base_prompt_text,
                    additional_instructions, glossary_content, character_dictionary,
                    block_pbar,
                    retry_count, retry_delay, request_delay
                )
                futures_map[future] = i

            for future in concurrent.futures.as_completed(futures_map):
                original_index = futures_map[future]
                try:
                    res_index, status, result_text, input_chars, output_chars = future.result()
                    if res_index != original_index: logging.error(f"Index mismatch! Map:{original_index}, Res:{res_index}"); continue

                    translation_data[res_index]["translated_text"] = result_text
                    translation_data[res_index]["status"] = status
                    total_input_chars += input_chars
                    total_output_chars += output_chars
                    logging.debug(f"Block {res_index} stage 1 status: {status}")

                    if status == 'success':
                        current_filepath = translation_data[res_index].get('filepath')
                        if isinstance(current_filepath, str):
                            try:
                                with open(current_filepath, 'w', encoding='utf-8') as f: f.write(result_text)
                                translated_file_count_ref[0] += 1
                            except Exception as write_e:
                                logging.error(f"Error writing success {current_filepath}: {write_e}")
                                translation_data[res_index]['status'] = 'failure'
                                if current_filepath not in failed_block_files: failed_block_files.append(current_filepath)
                        else:
                             translation_data[res_index]['status'] = 'failure'
                    elif status == 'needs_line_translation':
                        if res_index not in line_translation_candidates: line_translation_candidates.append(res_index)
                    elif status == 'failure':
                        current_filepath = translation_data[res_index].get('filepath')
                        if isinstance(current_filepath, str) and current_filepath not in failed_block_files: failed_block_files.append(current_filepath)

                    block_pbar.set_postfix({"번역 완료 텍스트 블록 수": f"{translated_file_count_ref[0]}/{total_files}"}, refresh=True)

                except Exception as e:
                    logging.error(f"Error future block index {original_index}: {e}", exc_info=True)
                    if 0 <= original_index < len(translation_data):
                         translation_data[original_index]['status'] = 'failure'
                         filepath_on_error = translation_data[original_index].get('filepath')
                         if isinstance(filepath_on_error, str) and filepath_on_error not in failed_block_files:
                              failed_block_files.append(filepath_on_error)

        tqdm.write("[번역 1 단계] 텍스트 블록 번역 완료")

    if line_translation_candidates:
        tqdm.write(f"\n[번역 2 단계] 번역 실패한 {len(line_translation_candidates)} 텍스트 블록 대상 줄 번역 시작")
        line_translation_candidates.sort()

        total_non_whitespace_input_line = sum(
            count_non_whitespace(translation_data[idx].get("current_text", ""))
            for idx in line_translation_candidates if 0 <= idx < len(translation_data)
        )

        total_lines_to_translate_stage2 = 0
        line_files_by_block_index = {}
        for index in line_translation_candidates:
             item = translation_data[index]
             filepath = item.get('filepath')
             if not filepath: continue
             try:
                 origin_backup_path = filepath.replace(".txt", ".origin.txt")
                 if os.path.exists(origin_backup_path): shutil.copy2(origin_backup_path, filepath)
                 else:
                     with open(filepath, 'w', encoding='utf-8') as f: f.write(item.get('current_text', ''))
                 _, line_files = split_text_block_for_initial_translation(filepath)
                 line_files_by_block_index[index] = line_files
                 total_lines_to_translate_stage2 += len(line_files)
             except Exception as split_err:
                  logging.error(f"Error prepare/split block {index} line count: {split_err}")
        tqdm.write(f"총 번역 대상 줄 수: {total_lines_to_translate_stage2}")

        completed_lines_count = [0]

        with tqdm(total=total_non_whitespace_input_line, desc="번역 진행률 ", unit=" 자", dynamic_ncols=True, position=0, leave=True) as line_pbar:
            for index in line_translation_candidates:
                item = translation_data[index]
                filepath = item.get('filepath')
                if not filepath: continue

                logging.info(f"Starting line fallback block index {index} ({filepath})")
                lines_all_success_flag = False
                try:
                    line_files = line_files_by_block_index.get(index)
                    if not line_files:
                         logging.error(f"No line files block {index}. Skip.")
                         translation_data[index]["status"] = 'failure'
                         if filepath not in failed_block_files: failed_block_files.append(filepath)
                         continue

                    line_order_json_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(filepath))[0]}.json")
                    line_trans_json_path = create_translation_json_for_lines(output_dir, line_files, filepath)

                    line_input_chars, line_output_chars, lines_all_success_flag = translate_lines(
                        output_dir, selected_model, api_key, temperature, top_p, top_k,
                        base_prompt_instructions, base_prompt_text, glossary_content,
                        line_pbar,
                        filepath,
                        total_lines_to_translate_stage2, # Pass total line count for postfix
                        completed_lines_count, # Pass completed line counter
                        character_dictionary, num_parallel=num_parallel, api_call_delay=request_delay
                    )
                    total_input_chars += line_input_chars
                    total_output_chars += line_output_chars

                    merge_translated_lines(output_dir, filepath, line_order_json_path)

                    if lines_all_success_flag:
                        translated_file_count_ref[0] += 1
                        #tqdm.write(f"Block {index} ({os.path.basename(filepath)}) OK via line. (Overall: {translated_file_count_ref[0]}/{total_files})")
                        try:
                            with open(filepath, 'r', encoding='utf-8') as f: final_translated_text = f.read()
                            translation_data[index]["translated_text"] = final_translated_text
                            translation_data[index]["status"] = 'success_via_lines'
                            if filepath in failed_block_files: failed_block_files.remove(filepath)
                        except Exception as read_final_e:
                             logging.error(f"Read error line success {filepath}: {read_final_e}")
                             translation_data[index]["status"] = 'failure_after_lines'
                             if filepath not in failed_block_files: failed_block_files.append(filepath)
                             translated_file_count_ref[0] -= 1
                    else:
                        logging.error(f"Line trans process failed block {index} ({filepath}).")
                        translation_data[index]["status"] = 'failure'
                        if filepath not in failed_block_files: failed_block_files.append(filepath)

                except Exception as line_proc_e:
                    logging.error(f"Error line fallback setup/cleanup block {index} ({filepath}): {line_proc_e}", exc_info=True)
                    translation_data[index]["status"] = 'failure'
                    if filepath not in failed_block_files: failed_block_files.append(filepath)

        tqdm.write("\n[번역 2 단계] 줄 번역 완료")
    else:
        print("\n[번역 2 단계] 줄 번역 대상 없음")

    try:
        with open(temp_json_path, 'w', encoding='utf-8') as f:
            json.dump(translation_data, f, indent=4, ensure_ascii=False)
        logging.info(f"Final translation data saved to {temp_json_path}")
    except Exception as e:
        logging.error(f"Failed save final data to {temp_json_path}: {e}")

    final_failed_files = sorted(list(set(failed_block_files)))
    print(f"\n1차 번역 완료. 전체 번역 성공 텍스트 블록: {translated_file_count_ref[0]}/{total_files}")
    if final_failed_files:
        print_colored(f"일부 번역 실패 텍스트 블록: {len(final_failed_files)}", colorama.Fore.RED)

    return final_failed_files, total_input_chars, total_output_chars

# --- 일본어 반복 문자 전처리 관련 코드 ---

# (JP_TO_KR_CHAR_MAP 딕셔너리는 제공된 것과 동일하게 유지합니다)
JP_TO_KR_CHAR_MAP = {
    # --- 청음 (清音) ---
    # 아 행 (あ行)
    'あ': '아', 'い': '이', 'う': '우', 'え': '에', 'お': '오',
    # 카 행 (か行)
    'か': '카', 'き': '키', 'く': '쿠', 'け': '케', 'こ': '코',
    # 사 행 (さ行)
    'さ': '사', 'し': '시', 'す': '스', 'せ': '세', 'そ': '소',
    # 타 행 (た行)
    'た': '타', 'ち': '치', 'つ': '츠', 'て': '테', 'と': '토',
    # 나 행 (な行)
    'な': '나', 'に': '니', 'ぬ': '누', 'ね': '네', 'の': '노',
    # 하 행 (は行)
    'は': '하', 'ひ': '히', 'ふ': '후', 'へ': '헤', 'ほ': '호',
    # 마 행 (ま行)
    'ま': '마', 'み': '미', 'む': '무', 'め': '메', 'も': '모',
    # 야 행 (や行)
    'や': '야', 'ゆ': '유', 'よ': '요',
    # 라 행 (ら行)
    'ら': '라', 'り': '리', 'る': '루', 'れ': '레', 'ろ': '로',
    # 와 행 (わ行)
    'わ': '와', 'ゐ': '위', 'ゑ': '웨', 'を': '오',
    # 응
    'ん': '응',

    # --- 탁음 (濁音) ---
    # 가 행 (が行)
    'が': '가', 'ぎ': '기', 'ぐ': '구', 'げ': '게', 'ご': '고',
    # 자 행 (ざ行)
    'ざ': '자', 'じ': '지', 'ず': '즈', 'ぜ': '제', 'ぞ': '조',
    # 다 행 (だ行)
    'だ': '다', 'ぢ': '지', 'づ': '즈', 'で': '데', 'ど': '도',
    # 바 행 (ば行)
    'ば': '바', 'び': '비', 'ぶ': '부', 'べ': '베', 'ぼ': '보',

    # --- 반탁음 (半濁音) ---
    # 파 행 (ぱ行)
    'ぱ': '파', 'ぴ': '피', 'ぷ': '푸', 'ぺ': '페', 'ぽ': '포',

    # --- 작은 히라가나 (촉음, 요음 등) ---
    'ぁ': '아', 'ぃ': '이', 'ぅ': '우', 'ぇ': '에', 'ぉ': '오',
    'ゃ': '야', 'ゅ': '유', 'ょ': '요',
    


    # ==========================================================================
    # 카타카나 (Katakana)
    # ==========================================================================

    # --- 청음 (清音) ---
    # 아 행 (ア行)
    'ア': '아', 'イ': '이', 'ウ': '우', 'エ': '에', 'オ': '오',
    # 카 행 (カ行)
    'カ': '카', 'キ': '키', 'ク': '쿠', 'ケ': '케', 'コ': '코',
    # 사 행 (サ行)
    'サ': '사', 'シ': '시', 'ス': '스', 'セ': '세', 'ソ': '소',
    # 타 행 (タ行)
    'タ': '타', 'チ': '치', 'ツ': '츠', 'テ': '테', 'ト': '토',
    # 나 행 (ナ行)
    'ナ': '나', 'ニ': '니', 'ヌ': '누', 'ネ': '네', 'ノ': '노',
    # 하 행 (ハ行)
    'ハ': '하', 'ヒ': '히', 'フ': '후', 'ヘ': '헤', 'ホ': '호',
    # 마 행 (マ行)
    'マ': '마', 'ミ': '미', 'ム': '무', 'メ': '메', 'モ': '모',
    # 야 행 (ヤ行)
    'ヤ': '야', 'ユ': '유', 'ヨ': '요',
    # 라 행 (ラ行)
    'ラ': '라', 'リ': '리', 'ル': '루', 'レ': '레', 'ロ': '로',
    # 와 행 (ワ行)
    'ワ': '와', 'ヰ': '위', 'ヱ': '웨', 'ヲ': '오',
    # 응
    'ン': '응',

    # --- 탁음 (濁音) ---
    # 가 행 (ガ行)
    'ガ': '가', 'ギ': '기', 'グ': '구', 'ゲ': '게', 'ゴ': '고',
    # 자 행 (ザ行)
    'ザ': '자', 'ジ': '지', 'ズ': '즈', 'ゼ': '제', 'ゾ': '조',
    # 다 행 (ダ行)
    'ダ': '다', 'ヂ': '지', 'ヅ': '즈', 'デ': '데', 'ド': '도',
    # 바 행 (バ行)
    'バ': '바', 'ビ': '비', 'ブ': '부', 'ベ': '베', 'ボ': '보',
    'ヴ': '부', # (외래어 v 발음 표기)

    # --- 반탁음 (半濁音) ---
    # 파 행 (パ行)
    'パ': '파', 'ピ': '피', 'プ': '푸', 'ペ': '페', 'ポ': '포',

    # --- 작은 카타카나 (촉음, 요음 등) ---
    'ァ': '아', 'ィ': '이', 'ゥ': '우', 'ェ': '에', 'ォ': '오',
    'ャ': '야', 'ュ': '유', 'ョ': '요',

}

def preprocess_repeated_japanese_chars(text: str) -> str:
    """
    일본어 텍스트에서 3회 이상 반복된 단일 문자를 해당 문자의 한국어 발음으로 치환합니다.
    예: 「おおおおお！！！」 -> 「오오오오오！！！」
    """
    if not text:
        return ""

    # 1. MAP의 모든 키(일본어 문자)를 합쳐 정규식에 사용할 문자셋을 만듭니다.
    #    re.escape()를 사용하여 정규식 특수문자를 안전하게 처리합니다.
    jp_chars_for_regex = ''.join(JP_TO_KR_CHAR_MAP.keys())
    pattern = f'([{re.escape(jp_chars_for_regex)}])\\1{{2,}}'

    # 2. 치환 로직을 담은 내부 함수를 정의합니다.
    def replacer(match: re.Match) -> str:
        # group(1)은 반복되는 한 개의 일본어 문자 (예: 'お')
        repeating_char_jp = match.group(1)
        # group(0)은 매치된 전체 문자열 (예: 'おおおおお')
        num_repeats = len(match.group(0))

        # 맵에서 해당 일본어 문자의 한국어 발음을 찾습니다.
        repeating_char_kr = JP_TO_KR_CHAR_MAP[repeating_char_jp]

        # 한국어 발음을 원래 횟수만큼 반복하여 반환합니다.
        return repeating_char_kr * num_repeats

    # 3. re.sub를 사용하여 텍스트 전체에서 패턴에 맞는 부분을 찾아 replacer 함수로 치환합니다.
    return re.sub(pattern, replacer, text)
    
    
# --- 일본어 반복 문자 전처리 관련 코드 끝 ---

# 기존 `translate_single_block` 함수 시작 (이 부분부터 기존 함수 내용으로 대체)
def translate_single_block(item, model, base_prompt_instructions, base_prompt_text, additional_instructions, glossary_content, character_dictionary, block_pbar, retry_count=3, retry_delay=10, api_call_delay=0.6):
    import copy
    item = copy.deepcopy(item)  # 병렬 환경에서 안전하게 복사

    filepath = item['filepath']
    index_in_list = item.get('item_index_in_list', -1)
    if index_in_list == -1:
        logging.error(f"Item index missing in translate_single_block for {filepath}. Returning -1.")
        return -1, 'failure', item.get('current_text', ''), 0, 0

    original_text = item.get("current_text", "")
    original_non_whitespace_input = count_non_whitespace(original_text)

    input_chars_attempted = 0
    final_output_chars = 0
    status = 'failure'
    pbar_non_whitespace_updated_block = 0
    stream_successful = False
    full_translated_text_from_stream = ""

    logging.debug(f"Worker started block {index_in_list} ({os.path.basename(filepath)})")

    # ✅ 전처리 한 번만 수행하여 item에 저장
    if "preprocessed_text" not in item:
        item["preprocessed_text"] = preprocess_repeated_japanese_chars(original_text)
        if original_text != item["preprocessed_text"]:
            logging.info(f"Block {index_in_list} ({os.path.basename(filepath)}) 반복 문자 전처리 적용됨")
        else:
            logging.debug(f"Block {index_in_list} 전처리 변화 없음.")

    for attempt in range(retry_count):
        try:
            if not isinstance(filepath, str):
                raise TypeError(f"Invalid filepath type: {type(filepath)}")

            prompt = create_prompt(
                base_prompt_instructions,
                base_prompt_text,
                additional_instructions,
                glossary_content,
                character_dictionary,
                {
                    "prev_context": item.get("prev_context", ""),
                    "current_text": item["preprocessed_text"]  # ✅ 항상 전처리된 텍스트 사용
                }
            )

            input_chars_attempted = len(prompt)

            if attempt > 0: time.sleep(retry_delay)
            if api_call_delay > 0: time.sleep(api_call_delay)

            logging.info(f"Streaming block attempt {attempt+1}/{retry_count} index {index_in_list}")

            response_stream = model.generate_content(prompt, stream=True)

            current_block_output_chars_this_attempt = 0
            full_translated_text_from_stream = ""

            for chunk in response_stream:
                chunk_text = ""
                try:
                    if hasattr(chunk, 'parts') and chunk.parts:
                        chunk_text = chunk.parts[0].text
                    elif hasattr(chunk, 'text'):
                        chunk_text = chunk.text
                    else:
                        continue
                except Exception as chunk_err:
                    logging.error(f"Chunk error block {index_in_list}: {chunk_err} - Chunk: {chunk}")
                    continue

                full_translated_text_from_stream += chunk_text
                current_block_output_chars_this_attempt += len(chunk_text)

                non_whitespace_chunk_chars = count_non_whitespace(chunk_text)
                if block_pbar and non_whitespace_chunk_chars > 0:
                    block_pbar.update(non_whitespace_chunk_chars)
                    pbar_non_whitespace_updated_block += non_whitespace_chunk_chars

            if not full_translated_text_from_stream.strip():
                logging.warning(f"Empty result stream block {index_in_list} (attempt {attempt+1}).")
                continue

            # ✅ 성공
            final_output_chars = current_block_output_chars_this_attempt
            status = 'success'
            stream_successful = True
            break

        except InvalidArgument as iae:
            logging.error(f"InvalidArgument API block {index_in_list} (attempt {attempt+1}): {iae}. Marking line.")
            status = 'needs_line_translation'
            break
        except TypeError as te:
            logging.error(f"TypeError worker block {index_in_list}: {te}. Item: {item}")
            status = 'failure'
            break
        except Exception as e:
            logging.warning(f"Error stream block {index_in_list} ({os.path.basename(filepath)}) attempt {attempt+1}/{retry_count}: {e}")
            continue

    processed_text = ""
    if status == 'success' and stream_successful:
        processed_text = apply_regex_transformations(full_translated_text_from_stream)
    elif status == 'needs_line_translation':
        processed_text = original_text
        final_output_chars = len(processed_text)
    else:
        logging.warning(f"텍스트 블록 번역 최종 실패: {os.path.basename(filepath)} -> 줄 번역 시도")
        status = 'needs_line_translation'
        processed_text = original_text
        final_output_chars = len(processed_text)

    if block_pbar:
        correction = original_non_whitespace_input - pbar_non_whitespace_updated_block
        if correction != 0:
            logging.debug(f"Applying final progress correction block {index_in_list} (Status: {status}): {correction}")
            block_pbar.update(correction)

    return index_in_list, status, processed_text, input_chars_attempted, final_output_chars

    
    
    
def translate_lines(output_dir, selected_model, api_key, temperature, top_p, top_k,
                    base_prompt_instructions, base_prompt_text, glossary_content,
                    line_pbar, # Pass the specific progress bar for lines
                    original_filepath,
                    total_lines_count, # Pass total number of lines for postfix
                    completed_lines_count, # Pass the reference list for completed lines
                    character_dictionary={}, num_parallel=3, api_call_delay=0.5):
    additional_instructions = load_prompt()
    base_filename = os.path.splitext(os.path.basename(original_filepath))[0]
    line_translation_json_path = os.path.join(output_dir, f"{base_filename}_line_trans.json")

    translation_data = []
    if os.path.exists(line_translation_json_path):
        try:
            with open(line_translation_json_path, 'r', encoding='utf-8') as f:
                translation_data = json.load(f)
        except json.JSONDecodeError as json_err:
            logging.error(f"Error decoding line JSON {line_translation_json_path}: {json_err}")
            return 0, 0, False
        except Exception as read_err:
            logging.error(f"Error reading line JSON {line_translation_json_path}: {read_err}")
            return 0, 0, False
    else:
        logging.error(f"Line translation JSON not found: {line_translation_json_path}.")
        return 0, 0, False

    if not translation_data:
         logging.warning(f"No lines in {line_translation_json_path} for {original_filepath}")
         return 0, 0, True

    try:
        client = genai.Client(api_key=api_key)
        model = GeminiModel(client, model_name=selected_model, safety_settings=safety_settings, generation_config={"temperature": temperature, "top_p": top_p, "top_k": top_k})
    except Exception as model_err:
        logging.error(f"Failed create model in translate_lines: {model_err}")
        return 0, 0, False

    total_lines_in_block = len(translation_data) # Use actual number of lines for this block
    total_input_chars_lines = 0
    total_output_chars_lines = 0

    prompt_details = {
        "base_instructions": base_prompt_instructions, "base_text": base_prompt_text,
        "additional_instructions": additional_instructions, "glossary": glossary_content,
        "characters": character_dictionary
    }

    logging.info(f"Streaming line translation for {os.path.basename(original_filepath)} ({total_lines_in_block} lines, max {num_parallel} workers)...")

    all_lines_successful = True
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_parallel) as executor:
        futures = [executor.submit(translate_single_line_item,
                                    item, i, model, prompt_details,
                                    line_pbar, # Pass line progress bar
                                    api_call_delay
                                   ) for i, item in enumerate(translation_data)]

        for future in concurrent.futures.as_completed(futures):
            try:
                idx, fpath, final_text_written, line_status, input_c, output_c = future.result()

                if idx < len(translation_data) and translation_data[idx]['filepath'] == fpath:
                    translation_data[idx]["translated_text"] = final_text_written
                    total_input_chars_lines += input_c
                    total_output_chars_lines += output_c

                    if line_status != 'success':
                         all_lines_successful = False

                else:
                    logging.warning(f"Index/filepath mismatch line {idx}. Block: {os.path.basename(original_filepath)}")
                    all_lines_successful = False

                # Update completed line count and postfix after each line finishes
                completed_lines_count[0] += 1
                if line_pbar: # Check if pbar object exists
                    line_pbar.set_postfix({"번역 완료 줄 수": f"{completed_lines_count[0]}/{total_lines_count}"}, refresh=True)

            except Exception as e:
                logging.error(f"Error processing line future for {original_filepath}: {e}", exc_info=True)
                all_lines_successful = False
                # Increment count even on future error, as the task slot is finished
                completed_lines_count[0] += 1
                if line_pbar:
                    line_pbar.set_postfix({"Completed lines": f"{completed_lines_count[0]}/{total_lines_count}"}, refresh=True)


    try:
        with open(line_translation_json_path, 'w', encoding='utf-8') as f:
            json.dump(translation_data, f, indent=4, ensure_ascii=False)
        logging.info(f"Updated line translation JSON saved: {line_translation_json_path}")
    except Exception as e:
        logging.error(f"Failed save updated line JSON {line_translation_json_path}: {e}")
        all_lines_successful = False

    logging.info(f"Streaming line translation finished for {os.path.basename(original_filepath)}.")

    return total_input_chars_lines, total_output_chars_lines, all_lines_successful


def translate_single_line_item(item, index, model, prompt_details, line_pbar, api_call_delay):
    filepath = item['filepath']
    current_text = ""
    try:
        with open(filepath, 'r', encoding='utf-8') as f: current_text = f.read()
    except Exception as read_err:
        logging.error(f"Read error line {filepath}: {read_err}")
        return index, filepath, "", 'read_failure', 0, 0

    original_non_whitespace_input_line = count_non_whitespace(current_text)

    prev_context = item.get("prev_context", "")
    input_chars = 0
    output_chars = 0
    text_to_write = current_text
    status = 'failure'
    pbar_non_whitespace_updated_line = 0
    stream_successful = False
    retry_attempts = 3

    try:
        prompt = create_prompt(
            prompt_details["base_instructions"], prompt_details["base_text"],
            prompt_details["additional_instructions"], prompt_details["glossary"],
            prompt_details["characters"],
            {"prev_context": prev_context, "current_text": current_text}
        )
        input_chars = len(prompt)

        for attempt in range(retry_attempts):
            try:
                if api_call_delay > 0: time.sleep(api_call_delay)
                logging.info(f"Streaming Line attempt {attempt+1} for {os.path.basename(filepath)}")

                response_stream = model.generate_content(prompt, stream=True)

                current_line_output_chars = 0
                current_full_translated_text = ""

                for chunk in response_stream:
                    chunk_text = ""
                    try:
                         if hasattr(chunk, 'parts') and chunk.parts: chunk_text = chunk.parts[0].text
                         elif hasattr(chunk, 'text'): chunk_text = chunk.text
                         else: continue
                    except Exception as chunk_err: logging.error(f"Chunk error line {index}: {chunk_err}"); continue

                    current_full_translated_text += chunk_text
                    current_line_output_chars += len(chunk_text)

                    non_whitespace_chunk_chars = count_non_whitespace(chunk_text)
                    if line_pbar and non_whitespace_chunk_chars > 0:
                        line_pbar.update(non_whitespace_chunk_chars)
                        pbar_non_whitespace_updated_line += non_whitespace_chunk_chars

                if not current_full_translated_text.strip():
                     logging.warning(f"Empty result stream line {index} (attempt {attempt+1}).")
                     if attempt < retry_attempts - 1: time.sleep(5)
                     continue

                text_to_write = apply_regex_transformations(
                                    postprocess_retranslated_line(current_full_translated_text)
                                )
                if not text_to_write.strip():
                    logging.warning(f"Result empty postprocess line {index} (attempt {attempt+1}).")
                    if attempt < retry_attempts - 1: time.sleep(5)
                    continue

                output_chars = len(text_to_write)
                status = 'success'
                stream_successful = True
                logging.info(f"Streaming line success for {os.path.basename(filepath)}")
                break

            except InvalidArgument as iae:
                 logging.error(f"InvalidArgument API Error line {index} (attempt {attempt+1}): {iae}. Marking failure.")
                 status = 'failure'; break
            except Exception as api_e:
                 logging.warning(f"API Error stream line {index} (attempt {attempt+1}): {api_e}")
                 if attempt < retry_attempts - 1: time.sleep(10)

        if not stream_successful:
            logging.warning(f"Streaming line failed for {os.path.basename(filepath)}. Keeping original.")
            text_to_write = current_text
            output_chars = len(text_to_write)
            status = 'failure'

            fail_msg = f"줄 번역 실패: {os.path.basename(filepath)}"
            if line_pbar: tqdm.write(f"{colorama.Fore.YELLOW}{fail_msg}{colorama.Style.RESET_ALL}")
            else: print_colored(fail_msg, colorama.Fore.YELLOW)

            if line_pbar and pbar_non_whitespace_updated_line > 0:
                 logging.info(f"Reverting progress failed line {index}: {-pbar_non_whitespace_updated_line}")
                 line_pbar.update(-pbar_non_whitespace_updated_line)
                 pbar_non_whitespace_updated_line = 0
        else:
             if line_pbar:
                 correction = original_non_whitespace_input_line - pbar_non_whitespace_updated_line
                 if correction != 0:
                     logging.debug(f"Applying final progress correction success line {index}: {correction}")
                     line_pbar.update(correction)

    except Exception as e:
        logging.error(f"Error processing line item {index} for {filepath}: {e}", exc_info=True)
        text_to_write = current_text
        output_chars = len(text_to_write)
        status = 'failure'
        fail_msg = f"Error processing line: {os.path.basename(filepath)} - {e}"
        if line_pbar: tqdm.write(f"{colorama.Fore.RED}{fail_msg}{colorama.Style.RESET_ALL}")
        else: print_colored(fail_msg, colorama.Fore.RED)


    try:
        with open(filepath, 'w', encoding='utf-8') as f: f.write(text_to_write)
        logging.debug(f"Written final content to line file {filepath}")
    except Exception as write_e:
         logging.error(f"Error writing final content to line file {filepath}: {write_e}")
         status = 'write_failure'

    return index, filepath, text_to_write, status, input_chars, output_chars


def retranslate_incomplete_blocks(output_dir, selected_model, api_key, temperature, top_p, top_k,
                                  base_prompt_instructions, glossary_content, character_dictionary,
                                  previous_context_number, retranslate_max_retries, num_parallel=5):
    print("\n\n[QC 2 단계] 텍스트 블록 별 시작/마지막 문장 비교 시작")
    try:
        client = genai.Client(api_key=api_key)
        model = GeminiModel(client, model_name=selected_model, safety_settings=safety_settings, generation_config={"temperature": temperature, "top_p": top_p, "top_k": top_k})
        additional_instructions = load_prompt()
        global base_prompt_text
        if 'base_prompt_text' not in globals(): raise NameError("'base_prompt_text' is not defined globally.")
    except NameError as ne:
        print_colored(f"Error: {ne}", colorama.Fore.RED); return 0, 0, 0
    except Exception as model_err:
        print_colored(f"Error creating model for similarity retranslation: {model_err}", colorama.Fore.RED); return 0, 0, 0

    temp_json_path = os.path.join(output_dir, "temp_translation.json")
    translation_data = []
    try:
        with open(temp_json_path, 'r', encoding='utf-8') as f:
            translation_data_original = json.load(f)
            translation_data = copy.deepcopy(translation_data_original)
    except Exception as e:
        print_colored(f"{temp_json_path} 로딩 에러: {e}. 재번역을 건너뜁니다.", colorama.Fore.RED, colorama.Style.BRIGHT)
        return 0, 0, 0

    prompt_details = {
        "base_instructions": base_prompt_instructions, "base_text": base_prompt_text,
        "additional_instructions": additional_instructions, "glossary": glossary_content,
        "characters": character_dictionary
    }

    total_input_chars_retranslate_all_retries = 0
    total_output_chars_retranslate_all_retries = 0
    estimated_retranslate_cost_all_retries = 0.0
    files_retranslated_in_previous_loop = []

    for retry_count in range(1, retranslate_max_retries + 1):
        print(f"\n{'=' * 20} 원본과 문장 유사도 비교 및 재번역 {retry_count}차 {'=' * 20}")
        files_to_check_this_loop = []
        files_to_retranslate_this_loop = []
        files_fixed_by_line_removal = []

        if retry_count == 1:
            print("비교 대상: 모든 텍스트 블록")
            for i, item in enumerate(translation_data):
                filepath = item.get('filepath')
                if not filepath: continue
                original_filepath = filepath.replace(".txt", ".origin.txt")
                if not os.path.exists(original_filepath): continue
                files_to_check_this_loop.append({"index": i, "filepath": filepath, "original_filepath": original_filepath})
        else:
            if not files_retranslated_in_previous_loop:
                print_colored("이전 차수에서 재번역된 텍스트 블록이 없습니다. 비교를 중지합니다.", colorama.Fore.GREEN, colorama.Style.BRIGHT)
                break
            print(f"비교 대상: 이전 차수 재번역 성공 {len(files_retranslated_in_previous_loop)} 파일")
            files_to_check_this_loop = files_retranslated_in_previous_loop

        if not files_to_check_this_loop:
             print_colored("비교 필요한 텍스트 블록이 없습니다.", colorama.Fore.YELLOW, colorama.Style.BRIGHT)
             break

        print("텍스트 블록 시작/마지막 문장 유사도 비교 중...")
        comparison_results = {}
        with tqdm(total=len(files_to_check_this_loop), desc=f"유사도 비교 {retry_count}차", unit="파일", dynamic_ncols=True, leave=False) as pbar_compare:
            for file_info in files_to_check_this_loop:
                i = file_info["index"]
                filepath = file_info["filepath"]
                original_filepath = file_info["original_filepath"]
                basename = os.path.basename(filepath)
                needs_retranslation = False
                final_status_message = "오류 발생"
                color = colorama.Fore.RED

                try:
                    with open(original_filepath, 'r', encoding='utf-8') as f_orig: original_text = f_orig.read()
                    with open(filepath, 'r', encoding='utf-8') as f_trans: translated_text = f_trans.read()
                    original_sentences = [s for s in original_text.splitlines() if s.strip()]
                    translated_sentences = [s for s in translated_text.splitlines() if s.strip()]

                    if not original_sentences or not translated_sentences:
                        final_status_message = "내용 없음, 건너뜁니다."
                        color = colorama.Fore.YELLOW + colorama.Style.BRIGHT
                    else:
                        first_sim, last_sim = compare_sentences_with_gemini(model, original_sentences[0], original_sentences[-1], translated_sentences[0], translated_sentences[-1], filepath)
                        if first_sim == "유사함" and last_sim == "유사함":
                            final_status_message = "재번역 불필요"
                            color = colorama.Fore.CYAN + colorama.Style.BRIGHT
                        elif first_sim == "유사하지 않음" and last_sim == "유사함":
                            fixed = try_fix_leading_lines(model, filepath, original_filepath, previous_context_number, translation_data, i)
                            if fixed:
                                final_status_message = "수정됨 (앞 줄 제거)"
                                color = colorama.Fore.GREEN + colorama.Style.BRIGHT
                                files_fixed_by_line_removal.append(file_info)
                            else:
                                final_status_message = "재번역 필요 (앞 줄 불일치, 수정 실패)"
                                color = colorama.Fore.MAGENTA + colorama.Style.BRIGHT
                                needs_retranslation = True
                        else:
                            reason = []
                            if first_sim != "유사함": reason.append("시작")
                            if last_sim != "유사함": reason.append("마지막")
                            if "오류" in first_sim or "오류" in last_sim: reason.append("유사도 확인 오류")
                            final_reason = "/".join(sorted(list(set(reason)))) + " 불일치"
                            final_status_message = f"재번역 필요 ({final_reason})"
                            color = colorama.Fore.MAGENTA + colorama.Style.BRIGHT
                            needs_retranslation = True

                    comparison_results[basename] = (final_status_message, color)

                    if needs_retranslation:
                        files_to_retranslate_this_loop.append(file_info)

                except FileNotFoundError:
                    comparison_results[basename] = ("파일 없음, 재번역 필요", colorama.Fore.YELLOW + colorama.Style.BRIGHT)
                    files_to_retranslate_this_loop.append(file_info)
                except Exception as e:
                    comparison_results[basename] = (f"비교 오류 ({e}), 재번역 필요", colorama.Fore.YELLOW + colorama.Style.BRIGHT)
                    files_to_retranslate_this_loop.append(file_info)
                finally: pbar_compare.update(1)

        print(f"\n{retry_count}차 유사도 비교 완료.")
        for file_info in files_to_check_this_loop:
             basename = os.path.basename(file_info['filepath'])
             msg, clr = comparison_results.get(basename, ("결과 없음", colorama.Fore.RED))
             print(f"{clr}- {basename}: {msg}{colorama.Style.RESET_ALL}")

        if not files_to_retranslate_this_loop:
            if files_fixed_by_line_removal:
                 print_colored(f"\n비교 결과 재번역 필요한 텍스트 블록이 없습니다 (일부 수정됨). ({retry_count}차).", colorama.Fore.GREEN, colorama.Style.BRIGHT)
            else:
                 print_colored(f"\n비교 결과 재번역 필요한 텍스트 블록이 없습니다. ({retry_count}차).", colorama.Fore.GREEN, colorama.Style.BRIGHT)
            files_retranslated_in_previous_loop = []
            break

        print(f"\n재번역 필요 텍스트 블록: {len(files_to_retranslate_this_loop)} ({retry_count}차)...")

        total_chars_to_retranslate_this_retry_qc2 = 0
        print("재번역 대상 원본 파일 글자 수 계산 중 (QC2)...")
        for file_info in files_to_retranslate_this_loop:
            try:
                with open(file_info["original_filepath"], 'r', encoding='utf-8') as f_orig:
                    total_chars_to_retranslate_this_retry_qc2 += count_non_whitespace(f_orig.read())
            except Exception as e:
                 logging.warning(f"재번역 대상 파일({file_info['original_filepath']}) 글자 수 계산 오류 (QC2): {e}")
        print(f"총 재번역 대상 글자 수: {total_chars_to_retranslate_this_retry_qc2}")

        current_retry_input_chars = 0
        current_retry_output_chars = 0
        current_retry_estimated_cost = 0.0
        successfully_retranslated_this_retry_qc2 = []
        total_files_this_loop_qc2 = len(files_to_retranslate_this_loop) # 현재 루프의 총 파일 수
        completed_files_count_qc2 = 0 # 현재 루프에서 완료된 파일 수

        with tqdm(total=total_chars_to_retranslate_this_retry_qc2, desc=f"재번역 {retry_count}차 진행률", unit="자", dynamic_ncols=True, position=0) as pbar_retranslate:
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_parallel) as executor:
                futures = {executor.submit(retranslate_single_block_streaming_parallel,
                                            file_info, model, translation_data, prompt_details, selected_model,
                                            pbar_retranslate): file_info
                           for file_info in files_to_retranslate_this_loop}

                for future in concurrent.futures.as_completed(futures):
                    original_file_info = futures[future]
                    try:
                        idx, success, input_c, output_c, cost, final_text = future.result()

                        current_retry_input_chars += input_c
                        current_retry_output_chars += output_c
                        current_retry_estimated_cost += cost

                        if success:
                            successfully_retranslated_this_retry_qc2.append(original_file_info)
                            translation_data[idx]["translated_text"] = final_text
                        else:
                            logging.error(f"Streaming parallel retranslation failed for index {idx} ({os.path.basename(original_file_info['filepath'])}), QC2")

                    except Exception as exc:
                        print_colored(f'\nError retranslating file {os.path.basename(original_file_info["filepath"])}: {exc}', colorama.Fore.RED)
                        logging.error(f"Exception during parallel retranslation future for {original_file_info['filepath']} (QC2): {exc}", exc_info=True)
                    finally:
                        # Increment completed count regardless of success/failure
                        completed_files_count_qc2 += 1
                        # Update postfix
                        pbar_retranslate.set_postfix({"재번역 파일": f"{completed_files_count_qc2}/{total_files_this_loop_qc2}"}, refresh=True)

        total_input_chars_retranslate_all_retries += current_retry_input_chars
        total_output_chars_retranslate_all_retries += current_retry_output_chars
        estimated_retranslate_cost_all_retries += current_retry_estimated_cost
        files_retranslated_in_previous_loop = successfully_retranslated_this_retry_qc2

        try:
            with open(temp_json_path, 'w', encoding='utf-8') as f:
                json.dump(translation_data, f, indent=4, ensure_ascii=False)
            logging.info(f"Updated {temp_json_path} after QC2 streaming retranslation attempt {retry_count}")
        except Exception as e:
             print_colored(f"Error updating {temp_json_path} after QC2 retranslation attempt {retry_count}: {e}", colorama.Fore.RED, colorama.Style.BRIGHT)

        print(f"{'=' * 20} 유사도 비교 및 재번역 {retry_count}차 종료 {'=' * 20}")

    print("\n[QC 2 단계] 텍스트 블록 별 시작/마지막 문장 비교 완료\n")
    return total_input_chars_retranslate_all_retries, total_output_chars_retranslate_all_retries, estimated_retranslate_cost_all_retries


def compare_sentences_with_gemini(model, original_first_sentence, original_last_sentence, translated_first_sentence, translated_last_sentence, filepath):
    """
    Gemini 모델을 사용하여 두 문장 쌍(시작, 마지막)의 의미 유사도를 개별적으로 비교하고,
    각 쌍의 유사도 판단 결과를 반환합니다.

    Args:
        model: 초기화된 Gemini 모델 객체.
        original_first_sentence (str): 원본 텍스트 블록의 첫 문장 (일본어).
        original_last_sentence (str): 원본 텍스트 블록의 마지막 문장 (일본어).
        translated_first_sentence (str): 번역된 텍스트 블록의 첫 문장 (한국어).
        translated_last_sentence (str): 번역된 텍스트 블록의 마지막 문장 (한국어).
        filepath (str): 처리 중인 파일 경로 (로깅 및 출력용).

    Returns:
        tuple[str, str]: (첫 문장 유사도, 마지막 문장 유사도).
                         각 값은 "유사함", "유사하지 않음", 또는 오류 상태 문자열일 수 있습니다.
    """
    # --- 프롬프트 수정 ---
    # 각 문장 쌍에 대해 개별적인 유사도 판단을 요청하고,
    # 지정된 형식으로 답변하도록 명확하게 지시합니다.
    prompt = f"""
1. 목표: 번역된 텍스트 블록의 시작과 끝 문장이 원문의 시작과 끝 문장과 일치하는지 확인합니다. 즉, 전체 텍스트 블록의 내용의 추가나 누락 없이 해당 블록의 분량이 제대로 번역되었는지 여부를 판단합니다.

2. 비교 기준: 원문과 번역문의 의미가 유사해야 합니다. 단순히 주어나 동사 등의 요소만 고려하는 것이 아니라, 전달하려는 정보가 유사한지 확인합니다. 번역 중 단어의 순서 변경이나 문맥을 고려한 일부 단어의 누락, 동의어 사용, 어투 변화 등이 발생할 수 있습니다. 이러한 변화는 허용되지만, 문장의 핵심 의미는 다르지 않아야 합니다.

3. 예외 처리: 번역 중 줄단위 병합 상황 고려: 번역 중 원문의 다수의 줄이 하나의 줄로 합쳐졌을 수 있습니다.
3-1) '시작 문장 쌍'에서 번역문의 내용이 원문보다 더 많을 경우:
 - 번역문의 시작 내용이 원문과 같으면 유사하다고 판단합니다. (동일 텍스트 블록 내 뒷 내용이 하나로 합쳐져서 내용이 많아진 것으로 판단)
 - 번역문의 시작 내용이 원문과 다르다면 유사하지 않다고 판단합니다. (이전 텍스트 블록의 마지막 내용이 번역문 앞에 더해진 것으로 판단)
3-2) '마지막 문장 쌍'에서 번역문의 내용이 원문보다 더 많을 경우:
 - 번역문 전체 내용을 확인하고, 원문의 내용이 번역문의 전체 내용 중에 *포함되어 있으면* 유사하다고 판단합니다. (동일 텍스트 블록 내 앞 내용이 하나로 합쳐져서 내용이 많아진 것으로 판단)
 
4. 고유 명사 처리: 번역 과정에서 일본 명사를 일본 발음으로 표시할 수 있음을 감안합니다. (예시: 원문-黄純, 번역문-키즈키)

5. 위 비교 기준과 예외 처리 기준을 적용하여 유사함과 유사하지 않음을 너그럽게 판단합니다. 동일한 문장 쌍을 가지고 여러번 판단 기회를 가지게 되었을 때 절반은 유사하다고 판단할 가능성이 있다면 유사함으로 판단합니다.

6. 답변 방법: 각 문장 쌍이 유사하다면 "유사함", 유사하지 않다면 "유사하지 않음"으로 답변해 주세요.
답변 형식은 아래 처럼 "시작 문장"과 "마지막 문장"에 대해 각각 "유사함"과 "유사하지 않음"으로 답변하고, 그 외 다른 설명은 추가하지 않습니다.

답변 형식 예시:
시작 문장: 유사함/유사하지 않음
마지막 문장: 유사함/유사하지 않음

#### 비교 할 문장 쌍 ####

텍스트 블록의 시작 문장 쌍:
원문 (일본어): "{original_first_sentence}"
번역문 (한국어): "{translated_first_sentence}"

텍스트 블록의 마지막 문장 쌍:
원문 (일본어): "{original_last_sentence}"
번역문 (한국어): "{translated_last_sentence}"
"""
    # --- 화면 출력 제거 ---
    # print(f"\n\n--- {os.path.basename(filepath)} ---\n")
    # print(f"첫 문장 (원문): {original_first_sentence}")
    # print(f"첫 문장 (번역): {translated_first_sentence}\n")
    # print(f"마지막 문장 (원문): {original_last_sentence}")
    # print(f"마지막 문장 (번역): {translated_last_sentence}\n")

    first_sentence_similarity = "API 호출 오류"
    last_sentence_similarity = "API 호출 오류"

    try:
        logging.info(f"Gemini Prompt for {os.path.basename(filepath)}:\n{prompt}")
        response = None
        for api_retry in range(3):
            try:
                response = model.generate_content(prompt)
                break
            except Exception as api_e:
                if api_retry < 2:
                    logging.warning(f"Gemini API call failed ({os.path.basename(filepath)}). Retrying in 5s ({api_retry+1}/3): {api_e}")
                    time.sleep(5)
                else: raise api_e

        if response is None: raise Exception("API call failed after retries")

        result = response.text.strip()
        logging.info(f"Gemini Raw Result for {os.path.basename(filepath)}:\n{result}")

        lines = result.splitlines()
        parse_error = False

        if len(lines) == 2:
            try:
                if lines[0].startswith("시작 문장:"):
                    first_sentence_similarity = lines[0].split(":", 1)[1].strip()
                    if first_sentence_similarity not in ["유사함", "유사하지 않음"]: parse_error = True; first_sentence_similarity = "형식 오류"
                else: parse_error = True; first_sentence_similarity = "형식 오류"

                if lines[1].startswith("마지막 문장:"):
                    last_sentence_similarity = lines[1].split(":", 1)[1].strip()
                    if last_sentence_similarity not in ["유사함", "유사하지 않음"]: parse_error = True; last_sentence_similarity = "형식 오류"
                else: parse_error = True; last_sentence_similarity = "형식 오류"

            except IndexError:
                logging.error(f"Error parsing Gemini response lines for {os.path.basename(filepath)}. Response: {result}")
                parse_error = True; first_sentence_similarity = "파싱 오류"; last_sentence_similarity = "파싱 오류"
        else:
            logging.warning(f"Unexpected lines ({len(lines)}) in Gemini response for {os.path.basename(filepath)}. Response: {result}")
            parse_error = True; first_sentence_similarity = "줄 수 오류"; last_sentence_similarity = "줄 수 오류"

        # --- 화면 출력 제거 ---
        # print(f"분석 결과:")
        # print(f"  시작 문장: {first_sentence_similarity}")
        # print(f"  마지막 문장: {last_sentence_similarity}")
        logging.info(f"Parsed Gemini Result for {os.path.basename(filepath)}: Start={first_sentence_similarity}, End={last_sentence_similarity}")

    except Exception as e:
        # --- 화면 출력 제거 ---
        # print_colored(f"Gemini API 호출 또는 처리 중 오류 발생 ({os.path.basename(filepath)}): {e}", colorama.Fore.RED, colorama.Style.BRIGHT)
        logging.error(f"Error Gemini API call/processing for {os.path.basename(filepath)}: {e}", exc_info=True)
        first_sentence_similarity = "API/처리 오류"
        last_sentence_similarity = "API/처리 오류"



def try_fix_leading_lines(model, filepath, original_filepath, previous_context_number, translation_data, index):
    """
    번역된 파일의 앞 줄을 제거하며 첫 문장 유사도를 다시 확인하여 파일을 수정 시도하고,
    성공 여부(True/False)를 반환합니다. (화면 출력 없음)
    """
    # --- 화면 출력 제거 ---
    # print_colored(f"-> {os.path.basename(filepath)}: 첫 문장 불일치, 마지막 문장 일치. 앞 줄 제거 시도 (최대 {previous_context_number}줄)...", colorama.Fore.CYAN)
    logging.info(f"Attempting to fix leading lines for {os.path.basename(filepath)} (max {previous_context_number} lines)")

    try:
        with open(original_filepath, 'r', encoding='utf-8') as f: original_text = f.read()
        with open(filepath, 'r', encoding='utf-8') as f: translated_text = f.read()

        original_lines = [line for line in original_text.splitlines() if line.strip()]
        translated_lines = [line for line in translated_text.splitlines() if line.strip()]

        if not original_lines or not translated_lines:
             logging.warning(f"Skip leading line fix {os.path.basename(filepath)}: empty content.")
             return False

        original_first = original_lines[0]
        original_last = original_lines[-1] # 재확인 시 사용

        for lines_to_remove in range(1, previous_context_number + 1):
            if lines_to_remove >= len(translated_lines):
                 logging.info(f"Stop leading line fix {os.path.basename(filepath)}: remove {lines_to_remove} lines leaves no content.")
                 break # 더 제거할 줄 없음

            modified_translated_lines = translated_lines[lines_to_remove:]
            modified_translated_text = "\n".join(modified_translated_lines)

            if not modified_translated_lines: # 제거 후 내용이 없으면 중단
                 logging.info(f"Stop leading line fix {os.path.basename(filepath)}: content empty after removing {lines_to_remove} lines.")
                 break

            temp_translated_first = modified_translated_lines[0]
            temp_translated_last = modified_translated_lines[-1]

            # --- 화면 출력 제거 ---
            # print_colored(f"   {lines_to_remove}줄 제거 후 첫 문장 유사도 재확인...", colorama.Fore.CYAN)
            logging.info(f"   Checking similarity after removing {lines_to_remove} lines from {os.path.basename(filepath)}")

            # 수정된 compare_sentences_with_gemini 호출
            first_sim_retry, last_sim_retry = compare_sentences_with_gemini(
                model, original_first, original_last, temp_translated_first, temp_translated_last, f"{filepath} (-{lines_to_remove} lines)"
            )

            # 첫 문장 유사도만 확인
            if first_sim_retry == "유사함":
                # --- 화면 출력 제거 ---
                # print_colored(f"   성공! {lines_to_remove}줄 제거 시 첫 문장 유사해짐. 파일 업데이트.", colorama.Fore.GREEN, colorama.Style.BRIGHT)
                logging.info(f"Success: Leading lines fixed {os.path.basename(filepath)} after removing {lines_to_remove} lines.")
                try:
                    with open(filepath, 'w', encoding='utf-8') as f: f.write(modified_translated_text)
                    translation_data[index]["translated_text"] = modified_translated_text
                    return True # 수정 성공
                except Exception as write_e:
                    logging.error(f"Error saving fixed content {os.path.basename(filepath)}: {write_e}", exc_info=True)
                    return False # 저장 실패 시 재번역 필요

            elif first_sim_retry == "유사하지 않음":
                # --- 화면 출력 제거 ---
                # print_colored(f"   {lines_to_remove}줄 제거 후에도 첫 문장 유사하지 않음.", colorama.Fore.CYAN)
                logging.info(f"   Still dissimilar after removing {lines_to_remove} lines from {os.path.basename(filepath)}.")
                # 다음 줄 수 제거로 계속
            else: # 오류 발생 시
                # --- 화면 출력 제거 ---
                # print_colored(f"   {lines_to_remove}줄 제거 후 유사도 확인 중 오류 발생 ({first_sim_retry}). 시도 중단.", colorama.Fore.YELLOW, colorama.Style.BRIGHT)
                logging.warning(f"   Error during similarity check after removing {lines_to_remove} lines from {os.path.basename(filepath)}: {first_sim_retry}. Stopping fix.")
                return False # 오류 시 재번역 필요

        # 루프 종료 (성공 못 함)
        # --- 화면 출력 제거 ---
        # print_colored(f"   최대 {previous_context_number}줄 제거 시도 후에도 첫 문장 불일치. 재번역 필요.", colorama.Fore.MAGENTA, colorama.Style.BRIGHT)
        logging.info(f"Failed to fix leading lines for {os.path.basename(filepath)} after trying {previous_context_number} removals.")
        return False # 수정 실패

    except FileNotFoundError:
        logging.error(f"FileNotFoundError during leading line fix for {os.path.basename(filepath)}.")
        return False # 재번역 필요
    except Exception as e:
        logging.error(f"Exception during leading line fix for {os.path.basename(filepath)}: {e}", exc_info=True)
        return False # 재번역 필요
        
        
      
def retranslate_by_line_count(output_dir, selected_model, api_key, temperature, top_p, top_k,
                              base_prompt_instructions, glossary_content, character_dictionary,
                              previous_context_number, retranslate_max_retries, translation_data, num_parallel=5):
    print("\n\n[QC 1 단계] 텍스트 블록 원본/번역본 문장 수 비교 시작")
    try:
        client = genai.Client(api_key=api_key)
        model = GeminiModel(client, model_name=selected_model, safety_settings=safety_settings, generation_config={"temperature": 1.5, "top_p": top_p, "top_k": top_k})
        additional_instructions = load_prompt()
        global base_prompt_text
        if 'base_prompt_text' not in globals(): raise NameError("'base_prompt_text' is not defined globally.")
    except NameError as ne:
        print_colored(f"Error: {ne}", colorama.Fore.RED); return 0, 0, 0
    except Exception as model_err:
        print_colored(f"Error creating model for line count retranslation: {model_err}", colorama.Fore.RED); return 0, 0, 0

    prompt_details = {
        "base_instructions": base_prompt_instructions, "base_text": base_prompt_text,
        "additional_instructions": additional_instructions, "glossary": glossary_content,
        "characters": character_dictionary
    }

    total_input_chars_retranslate_all_retries = 0
    total_output_chars_retranslate_all_retries = 0
    estimated_retranslate_cost_all_retries = 0.0
    files_retranslated_last_retry = []

    for retry_count in range(1, retranslate_max_retries + 1):
        print(f"\n{'=' * 20} 문장 수 비교 및 재번역 {retry_count}차 {'=' * 20}")
        files_to_check_this_retry = []
        files_to_retranslate_this_loop = []

        if retry_count == 1:
            print("비교 대상: 모든 텍스트 블록")
            for i, item in enumerate(translation_data):
                filepath = item.get('filepath')
                if not filepath: continue
                original_filepath = filepath.replace(".txt", ".origin.txt")
                if not os.path.exists(original_filepath): continue
                files_to_check_this_retry.append({
                    "index": i, "filepath": filepath, "original_filepath": original_filepath
                })
        else:
            if not files_retranslated_last_retry:
                print_colored("이전 차수에서 재번역된 파일이 없어 비교 및 재번역을 종료합니다.", colorama.Fore.GREEN, colorama.Style.BRIGHT)
                break
            print(f"비교 대상: 이전 차수 재번역 성공 {len(files_retranslated_last_retry)} 파일")
            files_to_check_this_retry = files_retranslated_last_retry

        if not files_to_check_this_retry:
             print_colored("비교할 파일이 없습니다.", colorama.Fore.YELLOW, colorama.Style.BRIGHT)
             break

        for file_info in files_to_check_this_retry:
            i = file_info["index"]
            filepath = file_info["filepath"]
            original_filepath = file_info["original_filepath"]
            try:
                with open(original_filepath, 'r', encoding='utf-8') as f_orig: original_text = f_orig.read()
                translated_filepath = filepath
                if not os.path.exists(translated_filepath):
                    print_colored(f"Warning: 번역 파일({os.path.basename(translated_filepath)})을 찾을 수 없어 비교를 건너뜁니다.", colorama.Fore.YELLOW, colorama.Style.BRIGHT)
                    continue
                with open(translated_filepath, 'r', encoding='utf-8') as f_trans: translated_text = f_trans.read()

                original_line_count = len([line for line in original_text.splitlines() if line.strip()])
                translated_line_count = len([line for line in translated_text.splitlines() if line.strip()])
                line_count_ratio = translated_line_count / original_line_count if original_line_count else 0
                retranslate_needed = not (0.7 <= line_count_ratio <= 1.3)

                color = colorama.Fore.MAGENTA + colorama.Style.BRIGHT if retranslate_needed else colorama.Fore.CYAN + colorama.Style.BRIGHT
                status_message = '재번역 필요' if retranslate_needed else '재번역 불필요'
                print(f"{color}- {os.path.basename(filepath)} (원본: {original_line_count}, 번역본: {translated_line_count}, 비율: {line_count_ratio:.2f}, {status_message}){colorama.Style.RESET_ALL}")

                if retranslate_needed:
                    file_info["original_line_count"] = original_line_count
                    file_info["translated_line_count"] = translated_line_count
                    file_info["line_count_ratio"] = line_count_ratio
                    files_to_retranslate_this_loop.append(file_info)
            except FileNotFoundError:
                print_colored(f"Warning: 비교 파일 ({os.path.basename(filepath)} 또는 원본)을 찾을 수 없습니다.", colorama.Fore.YELLOW, colorama.Style.BRIGHT)
            except Exception as e:
                print_colored(f"Warning: 파일 비교 중 오류 발생 ({os.path.basename(filepath)}): {e}", colorama.Fore.YELLOW, colorama.Style.BRIGHT)

        if not files_to_retranslate_this_loop:
            print_colored(f"\n비교 결과 재번역이 필요한 파일이 없습니다 ({retry_count}차).", colorama.Fore.GREEN, colorama.Style.BRIGHT)
            files_retranslated_last_retry = []
            break

        print(f"\n재번역 대상 파일 ({retry_count}차): {len(files_to_retranslate_this_loop)}개")
        for file_info in files_to_retranslate_this_loop:
             print(f"- {os.path.basename(file_info['filepath'])} (원본: {file_info.get('original_line_count', 'N/A')}, 번역본: {file_info.get('translated_line_count', 'N/A')}, 비율: {file_info.get('line_count_ratio', 0):.2f})")

        total_chars_to_retranslate_this_retry = 0
        print("재번역 대상 원본 파일 글자 수 계산 중...")
        for file_info in files_to_retranslate_this_loop:
            try:
                with open(file_info["original_filepath"], 'r', encoding='utf-8') as f_orig:
                    total_chars_to_retranslate_this_retry += count_non_whitespace(f_orig.read())
            except Exception as e:
                 logging.warning(f"재번역 대상 파일({file_info['original_filepath']}) 글자 수 계산 오류: {e}")
        print(f"총 재번역 대상 글자 수: {total_chars_to_retranslate_this_retry}")

        current_retry_input_chars = 0
        current_retry_output_chars = 0
        current_retry_estimated_cost = 0.0
        successfully_retranslated_this_retry = []
        total_files_this_loop = len(files_to_retranslate_this_loop) # 현재 루프의 총 파일 수
        completed_files_count = 0 # 현재 루프에서 완료된 파일 수

        with tqdm(total=total_chars_to_retranslate_this_retry, desc=f"재번역 {retry_count}차 진행률", unit="자", dynamic_ncols=True, position=0) as pbar_retranslate:
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_parallel) as executor:
                futures = {executor.submit(retranslate_single_block_streaming_parallel,
                                            file_info, model, translation_data, prompt_details, selected_model,
                                            pbar_retranslate): file_info
                           for file_info in files_to_retranslate_this_loop}

                for future in concurrent.futures.as_completed(futures):
                    original_file_info = futures[future]
                    try:
                        idx, success, input_c, output_c, cost, final_text = future.result()

                        current_retry_input_chars += input_c
                        current_retry_output_chars += output_c
                        current_retry_estimated_cost += cost

                        if success:
                            successfully_retranslated_this_retry.append(original_file_info)
                            translation_data[idx]["translated_text"] = final_text
                        else:
                            logging.error(f"Streaming parallel retranslation failed for index {idx} ({os.path.basename(original_file_info['filepath'])}), QC1")

                    except Exception as exc:
                        print_colored(f'\nError retranslating file {os.path.basename(original_file_info["filepath"])}: {exc}', colorama.Fore.RED)
                        logging.error(f"Exception during parallel retranslation future for {original_file_info['filepath']} (QC1): {exc}", exc_info=True)
                    finally:
                        # Increment completed count regardless of success/failure
                        completed_files_count += 1
                        # Update postfix
                        pbar_retranslate.set_postfix({"재번역 파일": f"{completed_files_count}/{total_files_this_loop}"}, refresh=True)


        total_input_chars_retranslate_all_retries += current_retry_input_chars
        total_output_chars_retranslate_all_retries += current_retry_output_chars
        estimated_retranslate_cost_all_retries += current_retry_estimated_cost
        files_retranslated_last_retry = successfully_retranslated_this_retry

        try:
            temp_json_path = os.path.join(output_dir, "temp_translation.json")
            with open(temp_json_path, 'w', encoding='utf-8') as f:
                json.dump(translation_data, f, indent=4, ensure_ascii=False)
            logging.info(f"Updated {temp_json_path} after QC1 streaming retranslation attempt {retry_count}")
        except Exception as e:
             print_colored(f"Error: {temp_json_path} 업데이트 중 오류 발생: {e}", colorama.Fore.RED, colorama.Style.BRIGHT)

        print(f"{'=' * 20} 문장 수 비교 및 재번역 {retry_count}차 종료 {'=' * 20}")

    print("\n[QC 1 단계] 텍스트 블록 원본/번역본 문장 수 비교 완료\n")
    return total_input_chars_retranslate_all_retries, total_output_chars_retranslate_all_retries, estimated_retranslate_cost_all_retries
    

def retranslate_single_block_parallel(file_info, model, translation_data, prompt_details, selected_model, api_retry_limit=3):
    """
    단일 텍스트 블록을 재번역하는 병렬 작업 함수 (QC 1 & 2 공통 사용).

    Args:
        file_info (dict): 재번역할 파일 정보 ({index, filepath, original_filepath}).
        model: Gemini 모델 객체.
        translation_data (list): 전체 번역 데이터 리스트 (업데이트 대상).
        prompt_details (dict): 프롬프트 생성을 위한 정보 ({base_instructions, base_text, ...}).
        selected_model (str): 모델 이름 (비용 계산용).
        api_retry_limit (int): 내부 API 호출 재시도 횟수.

    Returns:
        tuple: (index, success_flag, input_chars, output_chars, estimated_cost)
    """
    i = file_info["index"]
    filepath = file_info["filepath"]
    original_filepath = file_info["original_filepath"]
    success_flag = False
    input_chars_total = 0
    output_chars_total = 0
    estimated_cost_total = 0.0

    try:
        # 1. 원본 파일 내용으로 복원
        shutil.copy2(original_filepath, filepath)
    except Exception as e:
        print_colored(f"Error: 원본 파일 복원 실패 ({os.path.basename(filepath)}): {e}. 재번역 건너뜁니다.", colorama.Fore.RED, colorama.Style.BRIGHT)
        logging.error(f"Failed to restore original file for retranslation {filepath}: {e}")
        return i, False, 0, 0, 0.0 # 실패 반환

    # 2. 재번역 시도 (API 호출)
    for api_retry in range(api_retry_limit):
        try:
            # translation_data에서 원본 컨텍스트와 텍스트 가져오기
            item = translation_data[i]
            prompt = create_prompt(
                prompt_details["base_instructions"],
                prompt_details["base_text"],
                prompt_details["additional_instructions"],
                prompt_details["glossary"],
                prompt_details["characters"],
                {"prev_context": item["prev_context"],
                 "current_text": item["current_text"]} # 원본 텍스트 사용
            )

            current_input_chars = len(prompt)
            # input_chars_total += current_input_chars # 재시도마다 누적하지 않고 마지막 성공 기준으로 변경

            logging.info(f"Parallel Retranslation prompt for {os.path.basename(filepath)} (api_retry {api_retry + 1})")
            # 재번역 API 호출 (스트리밍 미사용 가정, 필요시 수정 가능)
            response = model.generate_content(prompt)
            translated_text = response.text
            # 후처리 적용
            processed_text = apply_regex_transformations(translated_text)
            logging.info(f"Parallel Retranslated text for {os.path.basename(filepath)}:\n{processed_text}")

            current_output_chars = len(processed_text)
            # output_chars_total = current_output_chars # 마지막 성공 기준으로 변경
            # estimated_cost_total = estimate_cost(current_input_chars, current_output_chars, selected_model) # 마지막 성공 기준으로 변경

            # 번역 결과 파일에 쓰기
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(processed_text)

            # translation_data 업데이트 (메모리상) - 중요: 이 부분은 스레드 안전
            translation_data[i]["translated_text"] = processed_text

            # 성공 시 정보 업데이트 및 플래그 설정 후 루프 종료
            input_chars_total = current_input_chars
            output_chars_total = current_output_chars           
            success_flag = True
            break # API 호출 성공 시 재시도 중단

        except Exception as e:
            last_error_reason = f"{type(e).__name__}"
            logging.warning(f"Parallel Retranslation API call failed for {filepath} (retry {api_retry + 1}/{api_retry_limit}): {e}")
            if api_retry < api_retry_limit - 1:
                # print_colored(f"Warning: API 호출 오류 ({os.path.basename(filepath)}). {10}초 후 재시도 ({api_retry + 1}/{api_retry_limit})", colorama.Fore.YELLOW, colorama.Style.BRIGHT)
                time.sleep(10) # 재시도 전 대기
                continue # 다음 재시도
            else: # 최종 실패
                print_colored(f"Error: API 호출 최종 실패 ({os.path.basename(filepath)} - {last_error_reason}): {e}", colorama.Fore.RED, colorama.Style.BRIGHT)
                # 최종 실패 시 원본 파일 내용으로 다시 복원 (이미 위에서 복원했으므로 추가 작업 불필요하나, 확실히 하려면 다시 복원)
                try:
                    shutil.copy2(original_filepath, filepath)
                    translation_data[i]["translated_text"] = translation_data[i]["current_text"] # 메모리도 원본으로
                except Exception as copy_err:
                    print_colored(f"Error: 재번역 최종 실패 후 원본 복원 중 오류 ({os.path.basename(filepath)}): {copy_err}", colorama.Fore.RED, colorama.Style.BRIGHT)
                success_flag = False
                break # API 호출 재시도 중단

    return i, success_flag, input_chars_total, output_chars_total, estimated_cost_total


def retranslate_single_block_streaming_parallel(file_info, model, translation_data, prompt_details, selected_model, pbar_retranslate, api_retry_limit=3):
    """
    (QC 1 & 2 용) 단일 텍스트 블록을 스트리밍으로 재번역하고 진행률을 업데이트하는 병렬 작업 함수.

    Args:
        file_info (dict): 재번역할 파일 정보 ({index, filepath, original_filepath}).
        model: Gemini 모델 객체.
        translation_data (list): 전체 번역 데이터 리스트 (컨텍스트 읽기용).
        prompt_details (dict): 프롬프트 생성을 위한 정보 ({base_instructions, base_text, ...}).
        selected_model (str): 모델 이름 (비용 계산용).
        pbar_retranslate (tqdm.tqdm): 업데이트할 진행률 바 객체.
        api_retry_limit (int): 내부 API 호출 재시도 횟수.

    Returns:
        tuple: (index, success_flag, input_chars, output_chars, estimated_cost, final_processed_text)
               final_processed_text: 성공 시 최종 번역 결과, 실패 시 원본 텍스트.
    """
    i = file_info["index"]
    filepath = file_info["filepath"]
    original_filepath = file_info["original_filepath"]
    success_flag = False
    input_chars_total = 0
    output_chars_total = 0
    estimated_cost_total = 0.0
    final_processed_text = ""
    original_text_content = "" # 원본 텍스트 저장용
    original_non_whitespace_input = 0 # 원본의 공백 제외 글자수 (진행률 보정용)

    try:
        # 1. 원본 파일 내용 읽기 (실패 시 재번역 불가)
        with open(original_filepath, 'r', encoding='utf-8') as f_orig:
            original_text_content = f_orig.read()
        final_processed_text = original_text_content # 기본값: 원본
        original_non_whitespace_input = count_non_whitespace(original_text_content)

        # 원본 파일 내용으로 작업 파일 복원 (재번역 전 상태 초기화)
        shutil.copy2(original_filepath, filepath)

    except Exception as e:
        print_colored(f"\nError: 재번역 위한 원본 파일 읽기/복원 실패 ({os.path.basename(filepath)}): {e}", colorama.Fore.RED)
        logging.error(f"Failed to read/restore original file for retranslation {filepath}: {e}", exc_info=True)
        # 오류 시에도 원본 글자수만큼 진행률을 채워야 함
        if pbar_retranslate and original_non_whitespace_input > 0:
            pbar_retranslate.update(original_non_whitespace_input)
        return i, False, 0, 0, 0.0, original_text_content # 실패 반환 (원본 텍스트 반환)

    pbar_non_whitespace_updated_retrans = 0 # 이번 재번역으로 pbar에 업데이트된 공백 제외 글자 수

    # 2. 재번역 시도 (API 호출 및 스트리밍)
    for api_retry in range(api_retry_limit):
        try:
            # translation_data에서 필요한 정보 가져오기 (컨텍스트 등)
            # 주의: translation_data는 다른 스레드에서 수정될 수 있으므로, 필요한 값만 읽어옴
            item_context = translation_data[i]["prev_context"] # 예시: 이전 문맥 읽기

            # 프롬프트 생성 (읽어온 원본 텍스트 사용)
            prompt = create_prompt(
                prompt_details["base_instructions"],
                prompt_details["base_text"],
                prompt_details["additional_instructions"],
                prompt_details["glossary"],
                prompt_details["characters"],
                {"prev_context": item_context,
                 "current_text": original_text_content} # 읽어온 원본 사용
            )

            current_input_chars = len(prompt)

            logging.info(f"Streaming Parallel Retranslation prompt for {os.path.basename(filepath)} (api_retry {api_retry + 1})")
            time.sleep(0.5) # 짧은 지연 추가 (API 동시 호출 완화)

            response_stream = model.generate_content(prompt, stream=True)

            current_full_translated_text = ""
            current_stream_output_chars = 0
            stream_successful_this_retry = False

            for chunk in response_stream:
                chunk_text = ""
                try:
                    if hasattr(chunk, 'parts') and chunk.parts: chunk_text = chunk.parts[0].text
                    elif hasattr(chunk, 'text'): chunk_text = chunk.text
                    else: continue
                except Exception as chunk_err:
                     logging.error(f"Chunk error during retranslation stream ({os.path.basename(filepath)}): {chunk_err}"); continue

                current_full_translated_text += chunk_text
                current_stream_output_chars += len(chunk_text)

                # 진행률 업데이트 (스트리밍)
                non_whitespace_chunk_chars = count_non_whitespace(chunk_text)
                if pbar_retranslate and non_whitespace_chunk_chars > 0:
                    pbar_retranslate.update(non_whitespace_chunk_chars)
                    pbar_non_whitespace_updated_retrans += non_whitespace_chunk_chars

            # 스트림 결과 확인
            if current_full_translated_text.strip():
                stream_successful_this_retry = True
            else:
                logging.warning(f"Empty result stream during retranslation {os.path.basename(filepath)} (attempt {api_retry+1}).")
                if api_retry < api_retry_limit - 1: time.sleep(5) # 다음 재시도 전 대기
                continue # 다음 재시도

            # 스트림 성공 시 후처리 및 결과 저장
            if stream_successful_this_retry:
                processed_text_attempt = apply_regex_transformations(current_full_translated_text)
                logging.info(f"Parallel Retranslated (streamed) text for {os.path.basename(filepath)}:\n{processed_text_attempt[:100]}...")

                # 파일에 쓰기 (재번역 결과)
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(processed_text_attempt)

                # 최종 결과 업데이트
                final_processed_text = processed_text_attempt
                input_chars_total = current_input_chars
                output_chars_total = len(final_processed_text)
                success_flag = True
                break # API 호출 및 처리 성공 시 재시도 중단

        except InvalidArgument as iae:
             last_error_reason = f"InvalidArgument: {iae}"
             logging.error(f"Parallel Retranslation API call InvalidArgument for {filepath} (retry {api_retry + 1}): {iae}. 재번역 실패 처리.")
             success_flag = False
             # InvalidArgument 는 재시도 의미 없을 수 있으므로 바로 중단
             break
        except Exception as e:
            last_error_reason = f"{type(e).__name__}: {e}"
            logging.warning(f"Parallel Retranslation API call failed for {filepath} (retry {api_retry + 1}/{api_retry_limit}): {e}\n{traceback.format_exc()}")
            if api_retry < api_retry_limit - 1:
                time.sleep(10) # 재시도 전 대기
                continue # 다음 재시도
            else: # 최종 실패
                print_colored(f"\nError: 재번역 API 호출 최종 실패 ({os.path.basename(filepath)} - {last_error_reason}). 원본 유지.", colorama.Fore.RED)
                success_flag = False
                break # API 호출 재시도 중단

    # 재번역 최종 실패 시, 원본 내용으로 복원 (이미 위에서 했지만 확인 차원)
    if not success_flag:
        try:
            # 실패 시 원본 내용이 final_processed_text 에 남아있어야 함
            if final_processed_text != original_text_content:
                final_processed_text = original_text_content
            # 파일 내용도 원본으로 확실히 되돌림
            shutil.copy2(original_filepath, filepath)
            logging.info(f"Retranslation failed for {filepath}, restored original content.")
        except Exception as restore_err:
            print_colored(f"Error: 재번역 실패 후 원본 복원 중 오류 ({os.path.basename(filepath)}): {restore_err}", colorama.Fore.RED)
            logging.error(f"Failed to restore original content after retranslation failure for {filepath}: {restore_err}")
        # 실패 시 input/output/cost는 0으로 반환
        input_chars_total = 0
        output_chars_total = 0
        estimated_cost_total = 0.0

    # 최종 진행률 보정 (성공/실패 모두 수행)
    # 재번역 대상 블록의 원본 글자수만큼 진행률이 채워지도록 보정
    if pbar_retranslate:
        correction = original_non_whitespace_input - pbar_non_whitespace_updated_retrans
        if correction != 0:
            logging.debug(f"Applying final progress correction retrans {os.path.basename(filepath)} (Status: {success_flag}): {correction}")
            pbar_retranslate.update(correction)

    return i, success_flag, input_chars_total, output_chars_total, estimated_cost_total, final_processed_text

    
def retranslate_text_blocks(output_dir, selected_model, api_key, temperature, top_p, top_k, total_input_chars, total_output_chars, retranslate_max_retries, previous_context_number, num_parallel=3):
    """텍스트 블록 중 일본어/한자가 남아있는 파일을 찾아 병렬로 재번역합니다."""
    retranslate_prompt_instruction = """
    * YOU ARE A TRANSLATION EXPERT PROFICIENT IN BOTH JAPANESE AND KOREAN.
     THE SENTENCES YOU WILL BE TRANSLATING ARE PARTIALLY TRANSLATED FROM JAPANESE TO KOREAN, BUT CONTAIN * REMAINING UNTRANSLATED JAPANESE TEXT OR KANJI.
     
    * YOUR TASK IS TO IDENTIFY THE JAPANESE OR KANJI SEGMENTS AND TRANSLATE THEM INTO NATURAL KOREAN, PRODUCING A COMPLETE KOREAN SENTENCE.
    
    * THE FINAL OUTPUT MUST BE A FULLY TRANSLATED, KOREAN SENTENCE, WITHOUT ANY JAPANESE REMAINING.
    * DO NOT INCLUDE ANY EXPLANATIONS, SUGGESTIONS, OR NOTES IN THE FINAL OUTPUT.
    
    * YOU MAY ONLY MODIFY THE EXISTING KOREAN TEXT IF DOING SO IS NECESSARY TO MAINTAIN FLUENCY AFTER INSERTING THE TRANSLATED JAPANESE SEGMENT.
    """
    retranslate_prompt_text = """
    PLEASE TRANSLATE THE FOLLOWING SENTENCE INTO KOREAN ACCORDING TO THE ABOVE INSTRUCTIONS. IF THE SENTENCE INCLUDES ANY PUNCTUATION MARKS, DO NOT OMIT THEM IN THE FINAL OUTPUT.

    ############### 번역 시작 ###############

    {text}  
    """

    retranslate_prompt = retranslate_prompt_instruction + "\n"
    try:
        with open("word_translation_instruction.txt", "r", encoding="utf-8") as f:
            word_translation_instruction = f.read().strip()
            if word_translation_instruction:
                retranslate_prompt += "[추가 지침]\n" + word_translation_instruction + "\n"
    except FileNotFoundError:
        print_colored("word_translation_instruction.txt 파일을 찾을 수 없습니다. 추가 지침 없이 재번역을 진행합니다.", colorama.Fore.YELLOW, colorama.Style.BRIGHT)

    retranslate_prompt += retranslate_prompt_text

    retry_count = 0
    total_input_chars_retranslate = 0
    total_output_chars_retranslate = 0
    estimated_retranslate_cost = 0

    print("\n\n[추가 번역] 문장 내 미번역된 일본어와 한자어 확인 시작\n")

    while retranslate_max_retries == 99 or retry_count < retranslate_max_retries:
        retry_count += 1
        print(f"\n{'=' * 20} 미번역 단어 재번역 {retry_count}차 {'=' * 20}")

        files_to_retranslate = []

        for filename in sorted(os.listdir(output_dir), key=lambda x: int(re.search(r'text_block_(\d+)', x).group(1)) if re.search(r'text_block_(\d+)', x) else float('inf')):
            if re.fullmatch(r"text_block_\d+\.txt", filename):
                filepath = os.path.join(output_dir, filename)
                if detect_japanese_or_chinese(open(filepath, 'r', encoding='utf-8').read()):
                    line_files = split_text_block_for_retranslation(filepath, retry_count)
                    for line_file in line_files:
                        with open(line_file, 'r', encoding='utf-8') as f:
                            if detect_japanese_or_chinese(f.read()):
                                files_to_retranslate.append(line_file)

        if not files_to_retranslate:
            print_colored("재번역할 파일이 없습니다.", colorama.Fore.GREEN, colorama.Style.BRIGHT)
            print("\n\n[문장 내 미번역된 일본어와 한자어 확인 종료]\n\n")
            merge_and_cleanup_retranslated_files(output_dir, retry_count)
            break

        files_to_retranslate.sort(key=lambda x: tuple(map(int, re.search(r'text_block_(\d+)_(\d+)\.txt', x).groups())) if re.search(r'text_block_(\d+)_(\d+)', x) else (float('inf'), ))
        retranslate_json_data = []
        for filepath in files_to_retranslate:
            with open(filepath, 'r', encoding='utf-8') as f:
                before_translate = f.read()
            retranslate_json_data.append({
                "filepath": filepath,
                "before_translate": before_translate,
                "after_translate": ""
            })

        retranslate_json_filepath = os.path.join(output_dir, f"retranslate_{retry_count}.json")
        with open(retranslate_json_filepath, 'w', encoding='utf-8') as f:
            json.dump(retranslate_json_data, f, indent=4, ensure_ascii=False)

        # --- 병렬 처리 ---
        input_chars_current_retry = 0
        output_chars_current_retry = 0
        client = genai.Client(api_key=api_key)
        model = GeminiModel(client, model_name=selected_model, safety_settings=safety_settings, generation_config={"temperature": 2, "top_p": top_p, "top_k": top_k})


        with tqdm(total=len(files_to_retranslate), desc=f"재번역 {retry_count}차 진행률", unit="파일", dynamic_ncols=True) as pbar_retranslate:
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_parallel) as executor:
                futures = [executor.submit(retranslate_single_line, filepath, selected_model, api_key, temperature, top_p, top_k, retranslate_prompt, retranslate_json_data, model) for filepath in files_to_retranslate] # model 전달
                for i, future in enumerate(concurrent.futures.as_completed(futures)):
                    try:
                        filepath, input_chars, output_chars = future.result()
                        input_chars_current_retry += input_chars
                        output_chars_current_retry += output_chars
                        pbar_retranslate.update(1)
                        pbar_retranslate.set_postfix({"재번역 파일 수": f"{i + 1}/{len(files_to_retranslate)}"})
                    except Exception as e:
                        print_colored(f"Error in retranslate_single_line: {e}", colorama.Fore.RED)


        total_input_chars_retranslate += input_chars_current_retry
        total_output_chars_retranslate += output_chars_current_retry

        merge_and_cleanup_retranslated_files(output_dir, retry_count)
        
    else: 
        print_colored(f"최대 재번역 횟수({retranslate_max_retries})에 도달하여 재번역을 중단합니다.", colorama.Fore.YELLOW, colorama.Style.BRIGHT)
        print("\n\n[추가 번역] 문장 내 미번역된 일본어와 한자어 확인 완료\n\n")
            
    total_input_chars += total_input_chars_retranslate   # 수정: 총 글자수 누적
    total_output_chars += total_output_chars_retranslate  # 수정: 총 글자수 누적
    return total_input_chars, total_output_chars, estimated_retranslate_cost   # 수정: 반환값 (글자 수)
    

def retranslate_single_line(filepath, selected_model, api_key, temperature, top_p, top_k, retranslate_prompt, retranslate_json_data, model): # model 추가
    """개별 줄을 재번역하고 결과를 반환합니다."""
    input_chars = 0
    output_chars = 0

    with open(filepath, 'r', encoding='utf-8') as f:
        original_line_content = f.read()

    if detect_japanese_or_chinese(original_line_content): # 일본어/한자 포함 여부 확인
        prompt = retranslate_prompt.format(text=original_line_content)
        logging.info(f"Retranslation prompt for {os.path.basename(filepath)}:\n{prompt}")

        try:
            input_chars = len(prompt)  # 입력 글자 수 계산
            response = model.generate_content(prompt) # model 객체 사용
            translated_text = response.text
            logging.info(f"Retranslated text for {os.path.basename(filepath)}:\n{translated_text}")
            translated_text = postprocess_retranslated_line(translated_text)  # 후처리 적용
            translated_text = apply_regex_transformations(translated_text) # 기본 후처리도 적용
            output_chars = len(translated_text)  # 출력 글자 수 계산

            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(translated_text)

            # JSON 데이터 업데이트
            for item in retranslate_json_data:
                 if item["filepath"] == filepath:
                     item["after_translate"] = translated_text
                     break

        except Exception as e:
            print_colored(f" 줄 단위 재번역 중 오류 발생: {filepath} - {e}", colorama.Fore.RED, colorama.Style.BRIGHT)
            with open(filepath, 'w', encoding='utf-8') as f: # 오류 시 원본 복원
                f.write(original_line_content)
    else:
        print_colored(f"{os.path.basename(filepath)}: 일본어/한자가 없으므로 재번역 건너뜁니다.", colorama.Fore.GREEN, colorama.Style.BRIGHT)

    return filepath, input_chars, output_chars  # 파일 경로, 입력/출력 글자 수 반환
    
    
def translate_lines_for_retranslate(line_files, selected_model, api_key, temperature, top_p, top_k, retranslate_prompt, num_parallel=3):
    """줄 단위 재번역을 병렬로 수행합니다."""
    client = genai.Client(api_key=api_key)
    generation_config = {
        "temperature": 2,
        "top_p": top_p,
        "top_k": top_k,
    }
    model = GeminiModel(client, model_name=selected_model, safety_settings=safety_settings, generation_config=generation_config) # model 객체 생성

    total_input_chars = 0
    total_output_chars = 0

    # --- ThreadPoolExecutor를 사용한 병렬 처리 ---
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_parallel) as executor:
        # future 객체 리스트 생성 (model 객체 전달)
        futures = [executor.submit(retranslate_single_line, filepath, selected_model, api_key, temperature, top_p, top_k, retranslate_prompt, [], model) for filepath in line_files]

        for future in concurrent.futures.as_completed(futures):
            try:
                filepath, input_chars, output_chars = future.result()
                total_input_chars += input_chars
                total_output_chars += output_chars

                # --- 후처리 적용 (병렬 처리 이후) ---
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                content = postprocess_retranslated_line(content) # 후처리 함수 호출
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content)
                # -------------------------------------

            except Exception as e:
                print_colored(f"Error in translate_lines_for_retranslate: {e}", colorama.Fore.RED)


    return total_input_chars, total_output_chars


def postprocess_retranslated_line(content):
    """재번역된 줄 단위 파일의 후처리 로직을 수행합니다."""

    # 1. 원문 남기고 번역문 추가하는 경우 처리
    match = re.match(r"(.+)\n(\n)?(.+)", content, re.DOTALL)
    if match:
        content = match.group(3)

    # 2. "->" 기호 처리
    content = content.split("->", 1)[-1].strip()
    
    # 3. 일본어 + 번역 괄호 패턴 처리 (예: '주민ごと(주민들까지)' → '주민들까지')
    content = re.sub(r'([\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+)\(([^)]+)\)', r'\2', content)

    return content
    
    
def merge_translated_lines(output_dir, original_filepath, line_json_filepath):
    line_data = []
    if os.path.exists(line_json_filepath):
        try:
            with open(line_json_filepath, 'r', encoding='utf-8') as f:
                line_data = json.load(f)
            logging.info(f"Loaded line order data from: {line_json_filepath}")
        except Exception as e:
            print_colored(f"Error: 순서 정보 파일({line_json_filepath}) 로드 오류: {e}", colorama.Fore.RED, colorama.Style.BRIGHT)
            logging.error(f"Failed to load or parse order JSON {line_json_filepath}: {e}", exc_info=True)
            return
    else:
        print_colored(f"Warning: 순서 정보 파일({line_json_filepath})이 없어 병합 불가.", colorama.Fore.YELLOW, colorama.Style.BRIGHT)
        logging.warning(f"Order JSON file not found, cannot merge: {line_json_filepath}")
        return

    merged_text = ""
    processed_line_files = []
    base_filename_for_lines = None

    json_basename = os.path.basename(line_json_filepath)
    # Use regex to robustly extract the base part (e.g., text_block_24 or text_block_24_2nd)
    match_2nd = re.match(r"(text_block_\d+)_2nd_order\.json$", json_basename)
    match_1st = re.match(r"(text_block_\d+)\.json$", json_basename) # Assuming 1st order file is just .json

    if match_2nd:
        base_filename_for_lines = match_2nd.group(1) + "_2nd" # e.g., text_block_24_2nd
    elif match_1st:
         # Check if it's the primary order file (not _line_trans.json)
         # This assumes the 1st fallback order file is named like 'text_block_N.json'
         # and the translation status json is '_line_trans.json'
         if not json_basename.endswith("_line_trans.json"):
              base_filename_for_lines = match_1st.group(1) # e.g., text_block_24

    # Handle potential case where naming doesn't match expectations
    if not base_filename_for_lines:
        # Fallback: Try deriving from original_filepath if possible, but log warning
        orig_basename = os.path.splitext(os.path.basename(original_filepath))[0]
        if orig_basename.startswith("text_block_"):
             base_filename_for_lines = orig_basename
             logging.warning(f"Could not determine line base filename from JSON path '{line_json_filepath}'. Falling back to derive from '{original_filepath}': using '{base_filename_for_lines}'.")
        else:
             logging.error(f"Could not determine base filename for lines from JSON path: {line_json_filepath} or original path: {original_filepath}")
             return

    for item in line_data:
        order = item.get('order')
        if order is None:
             logging.warning(f"Missing 'order' in {line_json_filepath} item: {item}. Skipping.")
             continue

        line_filename = os.path.join(output_dir, f"{base_filename_for_lines}_{order}.txt")
        processed_line_files.append(line_filename)

        try:
            with open(line_filename, 'r', encoding='utf-8') as line_file:
                content = line_file.read()
                merged_text += content + "\n"
        except FileNotFoundError:
            print_colored(f"Warning: 병합 대상 라인 파일({line_filename})이 없어 빈 줄로 대체.", colorama.Fore.YELLOW, colorama.Style.BRIGHT)
            logging.warning(f"Line file not found during merge: {line_filename}")
            merged_text += "\n"
        except Exception as e:
            print_colored(f"Error: 라인 파일({line_filename}) 읽기 오류: {e}", colorama.Fore.RED, colorama.Style.BRIGHT)
            logging.error(f"Error reading line file {line_filename} during merge: {e}")
            merged_text += "\n"

    try:
        with open(original_filepath, 'w', encoding='utf-8') as f:
            f.write(merged_text)
        logging.info(f"Merged lines into {original_filepath}")
    except Exception as e:
         print_colored(f"Error: 병합된 내용 쓰기 오류 ({original_filepath}): {e}", colorama.Fore.RED, colorama.Style.BRIGHT)
         logging.error(f"Error writing merged content to {original_filepath}: {e}")

    for line_filename in processed_line_files:
        if os.path.exists(line_filename):
            try:
                os.remove(line_filename)
            except Exception as e:
                print_colored(f"Error: 라인 파일({line_filename}) 삭제 오류: {e}", colorama.Fore.RED, colorama.Style.BRIGHT)
                logging.warning(f"Error removing line file {line_filename}: {e}")

    if os.path.exists(line_json_filepath):
        try:
            os.remove(line_json_filepath)
            logging.info(f"Removed order JSON file: {line_json_filepath}")
        except Exception as e:
            print_colored(f"Error: 순서 정보 파일({line_json_filepath}) 삭제 오류: {e}", colorama.Fore.RED, colorama.Style.BRIGHT)
            logging.warning(f"Error removing order JSON file {line_json_filepath}: {e}")
       

def merge_and_cleanup_retranslated_files(output_dir, retry_count):
    """
    모든 줄 단위 파일(text_block_*_*.txt)을 병합하고, 관련 JSON 파일도 삭제합니다.
    """
    for filename in os.listdir(output_dir):
        if filename.startswith("text_block_") and filename.endswith(".txt") and filename[-5:-4].isdigit():  # text_block_*.txt
            filepath = os.path.join(output_dir, filename)
            base_filename = os.path.splitext(os.path.basename(filepath))[0]

            # 해당 차수의 재번역 파일 목록 (정렬, retrans 없음) # 변경
            retrans_files = sorted([
                os.path.join(output_dir, f) for f in os.listdir(output_dir)
                if f.startswith(base_filename + "_") and f.endswith(".txt") and f != filename # 자기 자신 제외
            ], key=lambda x: int(re.search(r'_(\d+)\.txt', x).group(1))) # 정렬 기준 변경

            if retrans_files:  # 재번역 파일이 있으면 병합
                merged_retrans_text = ""
                for retrans_file in retrans_files:
                    try:
                        with open(retrans_file, 'r', encoding='utf-8') as f:
                            merged_retrans_text += f.read() + "\n"  # 줄바꿈 추가
                    except Exception as e:
                        print_colored(f"Error: 재번역 파일({retrans_file}) 읽기 오류: {e}", colorama.Fore.RED, colorama.Style.BRIGHT)
                        merged_retrans_text += "\n"

                # 원래 text_block_*.txt 파일에 병합된 내용 쓰기
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(merged_retrans_text)

                # 재번역 줄 단위 파일 삭제
                for retrans_file in retrans_files:
                    try:
                        os.remove(retrans_file)
                    except Exception as e:
                        print_colored(f"Error: 재번역 파일({retrans_file}) 삭제 오류: {e}", colorama.Fore.RED, colorama.Style.BRIGHT)    


def translate_single_file_2nd(filepath, model, prompt_details_2nd, pbar, api_call_delay=0.1, retry_attempts=3):
    """
    2차 번역을 위해 단일 파일을 처리하고 스트리밍 진행률을 업데이트하는 병렬 작업 함수.
    반드시 4개의 값을 반환합니다: original_char_count, input_chars, output_chars, status
    """
    original_text = ""
    original_char_count = 0
    original_non_whitespace_input_2nd = 0
    input_chars = 0
    output_chars = 0
    processed_text = ""
    status = 'failure' # 기본 상태: 실패
    pbar_non_whitespace_updated_2nd = 0

    try:
        # 1. 파일 읽기
        with open(filepath, 'r', encoding='utf-8') as f:
            original_text = f.read()
            original_char_count = len(original_text)
            original_non_whitespace_input_2nd = count_non_whitespace(original_text)
        processed_text = original_text # 기본값: 원본

        # 2. 2차 번역 프롬프트 생성
        prompt = create_prompt_2nd(
            prompt_details_2nd["additional_instructions"],
            prompt_details_2nd["glossary"],
            prompt_details_2nd["characters"],
            {"prev_context": "", "current_text": original_text}
        )
        input_chars_prompt = len(prompt) # 프롬프트 길이는 재시도와 무관하게 동일

        # 3. API 호출 (재시도 및 스트리밍 포함)
        stream_successful = False
        for retry in range(retry_attempts):
            try:
                # API 호출 지연 (유효성 검사 포함)
                is_delay_valid = False
                delay_value = 0.0
                if isinstance(api_call_delay, (int, float)):
                    try:
                        delay_value = float(api_call_delay)
                        if delay_value > 0:
                            is_delay_valid = True
                    except (ValueError, TypeError):
                        logging.warning(f"Invalid api_call_delay value: {api_call_delay}. Setting delay to 0.")
                        delay_value = 0.0
                elif api_call_delay is not None:
                     logging.warning(f"Unexpected type for api_call_delay: {type(api_call_delay)}. Value: {api_call_delay}. Setting delay to 0.")
                     delay_value = 0.0

                if is_delay_valid:
                    try:
                        time.sleep(delay_value)
                    except TypeError as sleep_err:
                        logging.error(f"TypeError during time.sleep({delay_value}): {sleep_err}")

                logging.info(f"Streaming 2nd Translation attempt {retry + 1} for {os.path.basename(filepath)}")

                # Gemini API 스트리밍 호출
                response_stream = model.generate_content(prompt, stream=True)

                current_full_translated_text = ""
                current_file_output_chars = 0

                for chunk in response_stream:
                    chunk_text = ""
                    try:
                        if hasattr(chunk, 'parts') and chunk.parts: chunk_text = chunk.parts[0].text
                        elif hasattr(chunk, 'text'): chunk_text = chunk.text
                        else: continue
                    except Exception as chunk_err:
                         logging.error(f"Chunk error 2nd trans {os.path.basename(filepath)}: {chunk_err} - Chunk: {chunk}"); continue

                    current_full_translated_text += chunk_text
                    current_file_output_chars += len(chunk_text)

                    # 진행률 업데이트
                    non_whitespace_chunk_chars = count_non_whitespace(chunk_text)
                    if pbar and non_whitespace_chunk_chars > 0:
                        pbar.update(non_whitespace_chunk_chars)
                        pbar_non_whitespace_updated_2nd += non_whitespace_chunk_chars

                # 스트림 결과 확인
                if not current_full_translated_text.strip():
                     logging.warning(f"Empty result stream 2nd trans {os.path.basename(filepath)} (attempt {retry+1}).")
                     if retry < retry_attempts - 1: time.sleep(5)
                     continue

                # 성공 시 후처리 및 결과 저장
                processed_text_attempt = apply_regex_transformations(current_full_translated_text)
                logging.info(f"2nd Translated processed text for {os.path.basename(filepath)} (attempt {retry+1}):\n{processed_text_attempt[:500]}...")

                processed_text = processed_text_attempt
                input_chars = input_chars_prompt # 성공 시 입력 글자 수 확정
                output_chars = len(processed_text) # 성공 시 출력 글자 수 확정
                status = 'success' # 성공 상태로 변경
                stream_successful = True
                logging.info(f"Streaming 2nd trans success for {os.path.basename(filepath)} (attempt {retry+1})")

                # 파일에 쓰기 (성공 시)
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(processed_text)

                break # 성공 시 재시도 중단

            except InvalidArgument as iae:
                 logging.error(f"InvalidArgument API Error during 2nd translation for {filepath} (retry {retry+1}): {iae}. Using original text.")
                 print_colored(f"\n오류: 2차 번역 API 호출 중 유효하지 않은 인자 ({os.path.basename(filepath)}): {iae}. 원본 유지.", colorama.Fore.RED)
                 processed_text = original_text
                 input_chars = input_chars_prompt # 입력은 발생했을 수 있음
                 output_chars = original_char_count # 출력은 원본 길이로
                 status = 'failure_invalid_arg' # 상태 업데이트
                 try:
                     with open(filepath, 'w', encoding='utf-8') as f: f.write(processed_text)
                 except Exception as write_e: logging.error(f"Error writing original text back after InvalidArgument for {filepath}: {write_e}")
                 stream_successful = False
                 break # 재시도 중단
            except Exception as e:
                logging.warning(f"API Error during 2nd translation for {filepath} (retry {retry+1}/{retry_attempts}): {e}")
                if retry < retry_attempts - 1:
                    time.sleep(10)
                else: # 최종 실패
                    print_colored(f"\nError: 2차 번역 API 최종 실패 ({os.path.basename(filepath)}): {e}. 원본 텍스트 사용.", colorama.Fore.RED, colorama.Style.BRIGHT)
                    processed_text = original_text
                    input_chars = input_chars_prompt
                    output_chars = original_char_count
                    status = 'failure_api' # 상태 업데이트
                    try:
                        with open(filepath, 'w', encoding='utf-8') as f: f.write(processed_text)
                    except Exception as write_e: logging.error(f"Error writing original text back after final API failure for {filepath}: {write_e}")
                    stream_successful = False
                    # break는 필요 없음 (마지막 재시도)

        # 최종 진행률 보정
        if pbar:
            correction = original_non_whitespace_input_2nd - pbar_non_whitespace_updated_2nd
            if correction != 0:
                logging.debug(f"Applying final progress correction 2nd trans {os.path.basename(filepath)} (Status: {status}): {correction}")
                pbar.update(correction)

    except Exception as file_proc_e:
         logging.error(f"Error processing file {filepath} in translate_single_file_2nd: {file_proc_e}", exc_info=True)
         print_colored(f"\nError: 2차 번역 파일 처리 오류 ({os.path.basename(filepath)}): {file_proc_e}", colorama.Fore.RED)
         original_char_count = 0
         input_chars = 0
         output_chars = 0
         status = 'failure_file_processing' # 상태 업데이트

    return original_char_count, input_chars, output_chars, status
    
    
def perform_second_translation(output_dir, selected_model, api_key, temperature, top_p, top_k,
                               glossary_content, character_dictionary, json_data, epub_file_path,
                               epub_dir, updated_opf_soup, updated_metadata, model,
                               cover_image_modify, cover_text_position, cover_text, font_path,
                               font_size, font_color, background_color, total_input_chars_1st,
                               total_output_chars_1st, start_time, translated_toc_map,
                               num_parallel, ridi_version):
    logging.info("Starting 2nd translation process...")

    epub_filename = json_data.get('epub_filename', 'unknown_epub')
    if epub_filename == 'unknown_epub':
         print_colored("Warning: json_data에서 epub_filename을 찾을 수 없습니다.", colorama.Fore.YELLOW)
         logging.warning("epub_filename not found in json_data during 2nd translation.")

    copied_files_count = 0
    text_block_pattern = re.compile(r"text_block_\d+\.txt$")
    origin_pattern = re.compile(r"\.origin\.txt$")
    second_pass_pattern = re.compile(r"_2nd\.txt$")

    for filename in os.listdir(output_dir):
        if text_block_pattern.search(filename) and \
           not origin_pattern.search(filename) and \
           not second_pass_pattern.search(filename):
            src_path = os.path.join(output_dir, filename)
            dest_path = os.path.join(output_dir, filename.replace(".txt", "_2nd.txt"))
            try:
                if os.path.isfile(src_path):
                    shutil.copy2(src_path, dest_path)
                    copied_files_count += 1
                else:
                    logging.warning(f"Source path is not a file, skipping copy: {src_path}")
            except Exception as copy_e:
                print_colored(f"Error: 2차 번역용 파일 복사 실패 ({filename}): {copy_e}", colorama.Fore.RED)
                logging.error(f"Failed to copy file for 2nd translation: {filename} - {copy_e}")
    logging.info(f"Copied {copied_files_count} text block files for 2nd translation.")

    json_filename = os.path.join(output_dir, epub_filename + ".json")
    json_2nd_filename = json_filename.replace(".json", "_2nd trans.json")
    json_data_2nd = None
    try:
        if not os.path.exists(json_filename):
             raise FileNotFoundError(f"Original JSON file not found: {json_filename}")

        shutil.copy2(json_filename, json_2nd_filename)
        with open(json_2nd_filename, 'r+', encoding='utf-8') as f:
            json_data_2nd = json.load(f)
            for xhtml_filename, xhtml_data in json_data_2nd.items():
                if xhtml_filename == "epub_filename": continue
                if isinstance(xhtml_data, list):
                    for block in xhtml_data:
                        if isinstance(block, dict) and block.get('type') == 'text_block':
                            original_content_path = block.get('content')
                            if original_content_path and isinstance(original_content_path, str):
                                 if not second_pass_pattern.search(original_content_path) and \
                                    not origin_pattern.search(original_content_path):
                                      block['content'] = original_content_path.replace(".txt", "_2nd.txt")
                            else:
                                 logging.warning(f"Invalid or missing content path in 2nd trans JSON for block in {xhtml_filename}: {block}")
                else:
                     logging.warning(f"Unexpected data structure for key {xhtml_filename} in 2nd trans JSON (expected list): {type(xhtml_data)}")


            f.seek(0)
            json.dump(json_data_2nd, f, indent=4, ensure_ascii=False)
            f.truncate()
        logging.info(f"Created 2nd translation JSON: {os.path.basename(json_2nd_filename)}")
    except FileNotFoundError as fnf_err:
         print_colored(f"Error: JSON 파일 처리 오류 - {fnf_err}", colorama.Fore.RED)
         logging.error(f"JSON file processing error: {fnf_err}")
         return None, None, None
    except Exception as json_e:
         print_colored(f"Error: 2차 번역용 JSON 파일 처리 중 오류 발생: {json_e}", colorama.Fore.RED)
         logging.error(f"Error processing 2nd translation JSON: {json_e}", exc_info=True)
         return None, None, None

    if json_data_2nd is None:
         print_colored("Error: 2차 번역 JSON 데이터 로드/생성 실패.", colorama.Fore.RED)
         logging.error("json_data_2nd is None after attempting to load/create.")
         return None, None, None

    additional_instructions = load_prompt()

    total_input_chars_2nd = 0
    total_output_chars_2nd = 0
    successful_2nd_translations = 0

    files_to_translate_2nd = []
    total_non_whitespace_chars_2nd = 0
    for filename in os.listdir(output_dir):
        if filename.startswith("text_block_") and filename.endswith("_2nd.txt"):
            filepath = os.path.join(output_dir, filename)
            files_to_translate_2nd.append(filepath)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    file_content = f.read()
                    total_non_whitespace_chars_2nd += count_non_whitespace(file_content)
            except Exception as read_e:
                print_colored(f"Warning: 파일 크기 계산 중 오류 ({filename}): {read_e}", colorama.Fore.YELLOW)
                logging.warning(f"Error calculating size for {filename}: {read_e}")

    def get_block_number_2nd(filepath):
        match = re.search(r'text_block_(\d+)_2nd\.txt$', os.path.basename(filepath))
        return int(match.group(1)) if match else float('inf')

    files_to_translate_2nd.sort(key=get_block_number_2nd)

    total_files = len(files_to_translate_2nd)
    print(f"총 {total_files}개의 텍스트 블록을 2차 번역합니다 (총 글자 수: {total_non_whitespace_chars_2nd}, 병렬 번역 수={num_parallel}).")
    logging.info(f"Starting parallel 2nd translation for {total_files} files (Total Non-WS Chars: {total_non_whitespace_chars_2nd}, Workers: {num_parallel}).")

    prompt_details_2nd = {
        "additional_instructions": additional_instructions,
        "glossary": glossary_content,
        "characters": character_dictionary
    }
    api_call_delay_2nd = 0.1

    if not model:
        print_colored("Error: 유효한 Gemini 모델 객체가 없어 2차 번역을 진행할 수 없습니다.", colorama.Fore.RED)
        logging.error("No valid Gemini model object available for 2nd translation.")
        return json_data_2nd, json_2nd_filename, None

    failed_2nd_block_indices = []
    block_statuses = {}

    with tqdm(total=total_non_whitespace_chars_2nd, desc="2차 번역 진행률", unit="자", dynamic_ncols=True, position=0, leave=True) as pbar_2nd_translate:
        translated_file_count = 0
        futures_map_2nd = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_parallel) as executor:
            for i, filepath in enumerate(files_to_translate_2nd):
                future = executor.submit(translate_single_file_2nd,
                                         filepath, model, prompt_details_2nd,
                                         pbar_2nd_translate, api_call_delay_2nd)
                futures_map_2nd[future] = i

            for future in concurrent.futures.as_completed(futures_map_2nd):
                original_index = futures_map_2nd[future]
                try:
                    original_chars, input_c, output_c, status = future.result()
                    total_input_chars_2nd += input_c
                    total_output_chars_2nd += output_c
                    block_statuses[original_index] = status

                    if status.startswith('success'):
                        successful_2nd_translations += 1
                    else:
                        if original_index not in failed_2nd_block_indices:
                            failed_2nd_block_indices.append(original_index)
                        logging.warning(f"2nd translation block failed (index: {original_index}, file: {os.path.basename(files_to_translate_2nd[original_index])}), status: {status}. Marking for fallback.")

                    translated_file_count += 1
                    pbar_2nd_translate.set_postfix({"번역 파일 수": f"{translated_file_count}/{total_files}"}, refresh=True)

                except Exception as e:
                    original_index = futures_map_2nd[future]
                    logging.error(f"Error processing 2nd translation future for index {original_index}: {e}", exc_info=True)
                    print_colored(f"\nError: 2차 번역 작업 처리 중 오류 발생 (파일 인덱스: {original_index}): {e}", colorama.Fore.RED)
                    if original_index not in failed_2nd_block_indices:
                        failed_2nd_block_indices.append(original_index)
                    block_statuses[original_index] = 'failure_future_exception'
                    translated_file_count += 1
                    pbar_2nd_translate.set_postfix({"번역 파일 수": f"{translated_file_count}/{total_files}"}, refresh=True)

    if failed_2nd_block_indices:
        print_colored(f"\n[2차 번역 Fallback] {len(failed_2nd_block_indices)}개 블록 줄 단위 처리 시작...", colorama.Fore.YELLOW)
        failed_2nd_block_indices.sort()

        total_chars_in_fallback = 0
        total_lines_in_fallback = 0
        line_files_by_block_index = {}

        print_colored("  Fallback 대상 파일 분할 및 크기 계산 중...", colorama.Fore.CYAN)
        for index in failed_2nd_block_indices:
            filepath_2nd = files_to_translate_2nd[index]
            line_order_json_path, line_files = split_text_block_2nd(filepath_2nd)
            if line_order_json_path and line_files:
                line_files_by_block_index[index] = {'order_json': line_order_json_path, 'lines': line_files}
                total_lines_in_fallback += len(line_files)
                for lf in line_files:
                    try:
                        with open(lf, 'r', encoding='utf-8') as f_line:
                            total_chars_in_fallback += count_non_whitespace(f_line.read())
                    except Exception: pass
            else:
                 logging.error(f"Failed to split block for 2nd fallback: {filepath_2nd}")
                 block_statuses[index] = 'failure_split_2nd'

        print_colored(f"  총 {total_lines_in_fallback}개 줄 Fallback 번역 시작 (총 글자 수: {total_chars_in_fallback}).", colorama.Fore.CYAN)

        with tqdm(total=total_chars_in_fallback, desc="2차 Fallback 진행률", unit="자", dynamic_ncols=True, position=0, leave=True) as pbar_2nd_fallback:
            completed_lines_count_ref = [0]

            for index in failed_2nd_block_indices:
                 if index not in line_files_by_block_index: continue

                 block_info = line_files_by_block_index[index]
                 line_files = block_info['lines']
                 line_order_json_path = block_info['order_json']
                 filepath_2nd = files_to_translate_2nd[index]

                 logging.info(f"Starting 2nd fallback for block index {index} ({os.path.basename(filepath_2nd)})")

                 input_c_lines, output_c_lines, all_lines_ok = translate_lines_2nd(
                     output_dir, line_files, model, prompt_details_2nd, pbar_2nd_fallback,
                     api_call_delay_2nd, num_parallel,
                     total_lines_in_fallback, completed_lines_count_ref
                 )
                 total_input_chars_2nd += input_c_lines
                 total_output_chars_2nd += output_c_lines

                 try:
                    merge_translated_lines(output_dir, filepath_2nd, line_order_json_path)
                    if all_lines_ok:
                         block_statuses[index] = 'success_via_lines_2nd'
                         logging.info(f"2nd fallback successful for block index {index}")
                    else:
                         block_statuses[index] = 'failure_lines_2nd'
                         logging.warning(f"2nd fallback completed but some lines failed for block index {index}")
                 except Exception as merge_err:
                      logging.error(f"Error merging 2nd fallback lines for index {index}: {merge_err}", exc_info=True)
                      block_statuses[index] = 'failure_merge_2nd'

        print_colored("[2차 번역 Fallback] 줄 단위 처리 완료.", colorama.Fore.YELLOW)

    else:
        print_colored("\n[2차 번역 Fallback] 블록 단위 처리 모두 성공, Fallback 불필요.", colorama.Fore.GREEN)


    final_successful_count = sum(1 for status in block_statuses.values() if status.startswith('success'))

    print(f"\n2차 번역 완료. (최종 성공: {final_successful_count}/{total_files} 파일)")
    logging.info(f"Finished 2nd translation (with fallback). Final Successful: {final_successful_count}/{total_files}. API Input Chars: {total_input_chars_2nd}, API Output Chars: {total_output_chars_2nd}")

    returned_path_standard_2nd = None
    returned_path_ridi_2nd = None

    epub_filename_base_2nd = os.path.splitext(epub_filename)[0]
    epub_suffix_standard_2nd = "_ko_2nd trans.epub"
    epub_filename_standard_2nd = f"{epub_filename_base_2nd}{epub_suffix_standard_2nd}"
    mod_epub_path_standard_2nd = os.path.join(epub_dir, epub_filename_standard_2nd)
    counter_std = 1
    while os.path.exists(mod_epub_path_standard_2nd):
        mod_epub_path_standard_2nd = os.path.join(epub_dir, f"{epub_filename_base_2nd}{epub_suffix_standard_2nd.replace('.epub', '')} ({counter_std}).epub")
        counter_std += 1

    print_colored("\n--- 2차 표준 번역 EPUB 생성 시작 ---", colorama.Fore.YELLOW)
    returned_path_standard_2nd = rebuild_epub_orchestrator(
        epub_path=epub_file_path,
        json_data=json_data_2nd,
        updated_opf_soup=updated_opf_soup,
        updated_metadata=updated_metadata,
        model=model,
        cover_image_modify=cover_image_modify,
        cover_text_position=cover_text_position,
        cover_text=cover_text,
        font_path=font_path,
        font_size=font_size,
        font_color=font_color,
        background_color=background_color,
        translated_toc_map=translated_toc_map,
        output_epub_path=mod_epub_path_standard_2nd,
        mode='standard'
    )

    if returned_path_standard_2nd:
         print_colored("2차 표준 번역 EPUB 생성 성공!", colorama.Fore.GREEN, colorama.Style.BRIGHT)

         if ridi_version == 1:
             print_colored("\n--- 2차 RIDI 버전 EPUB 생성 시작 ---", colorama.Fore.YELLOW)

             original_is_nav_doc = False
             original_is_ncx = False
             original_nav_doc_temp_path = None
             original_ncx_temp_path = None

             for epub_path_key, temp_path_val in translated_toc_map.items():
                 if epub_path_key.lower().endswith(('.xhtml', '.html')) and \
                    ('nav' in epub_path_key.lower() or 'toc' in epub_path_key.lower()):
                     if os.path.exists(temp_path_val):
                         original_is_nav_doc = True
                         original_nav_doc_temp_path = temp_path_val
                         break
                 elif epub_path_key.lower().endswith('.ncx'):
                      if os.path.exists(temp_path_val):
                          original_is_ncx = True
                          original_ncx_temp_path = temp_path_val

             if original_is_nav_doc:
                 original_is_ncx = False

             generated_ncx_path_2nd = None
             final_toc_files_map_for_ridi_2nd = {}

             if original_is_nav_doc:
                 print_colored("RIDI 버전용 NCX 변환 시도 중 (2차, Nav 원본)...", colorama.Fore.CYAN)
                 if original_nav_doc_temp_path:
                     try:
                         with open(original_nav_doc_temp_path, 'r', encoding='utf-8') as f_nav:
                             nav_content_str_2nd = f_nav.read()
                         ncx_bytes_generated_2nd = convert_nav_html_to_ncx(nav_content_str_2nd, updated_metadata)

                         if ncx_bytes_generated_2nd:
                             generated_ncx_path_2nd = os.path.join(output_dir, "generated_toc_2nd.ncx")
                             with open(generated_ncx_path_2nd, 'wb') as f_ncx:
                                 f_ncx.write(ncx_bytes_generated_2nd)
                             final_toc_files_map_for_ridi_2nd['toc.ncx'] = generated_ncx_path_2nd
                             print_colored("RIDI 버전용 NCX 파일 생성 성공 (2차).", colorama.Fore.GREEN)
                             logging.info(f"2차 RIDI EPUB용 목차 맵: 생성된 NCX 파일 사용 ({generated_ncx_path_2nd})")
                         else:
                             print_colored("경고: RIDI 버전용 NCX 파일 변환 실패 (2차).", colorama.Fore.YELLOW)
                             logging.warning("convert_nav_html_to_ncx (2nd pass) returned None.")
                             final_toc_files_map_for_ridi_2nd = {}
                     except FileNotFoundError:
                         print_colored(f"오류: 2차 RIDI용 Nav Doc 파일({original_nav_doc_temp_path})을 읽을 수 없습니다.", colorama.Fore.RED)
                         logging.error(f"Could not read Nav Doc file for 2nd pass NCX conversion: {original_nav_doc_temp_path}")
                         final_toc_files_map_for_ridi_2nd = {}
                     except Exception as conv_err_2nd:
                         print_colored(f"오류: RIDI 버전용 NCX 변환 중 예외 발생 (2차): {conv_err_2nd}", colorama.Fore.RED)
                         logging.error(f"NCX conversion exception (2nd pass): {conv_err_2nd}", exc_info=True)
                         final_toc_files_map_for_ridi_2nd = {}
                 else:
                      print_colored("경고: 2차 RIDI 생성에 사용할 Nav Doc 파일을 찾지 못했습니다.", colorama.Fore.YELLOW)
                      final_toc_files_map_for_ridi_2nd = {}

             elif original_is_ncx:
                 if original_ncx_temp_path:
                     final_toc_files_map_for_ridi_2nd['toc.ncx'] = original_ncx_temp_path
                     logging.info(f"2차 RIDI EPUB용 목차 맵: 1차 번역 NCX 사용 ({original_ncx_temp_path})")
                 else:
                     logging.error("2차 RIDI EPUB 생성 오류: 원본이 NCX였으나 1차 번역된 NCX 파일을 찾을 수 없습니다.")
                     final_toc_files_map_for_ridi_2nd = {}

             else:
                  logging.error("2차 RIDI EPUB 생성 오류: 사용할 수 있는 목차 파일(Nav Doc 또는 NCX)을 찾지 못했습니다.")
                  final_toc_files_map_for_ridi_2nd = {}

             epub_suffix_ridi_2nd = "_ko_2nd trans_RIDI.epub"
             epub_filename_ridi_2nd = f"{epub_filename_base_2nd}{epub_suffix_ridi_2nd}"
             mod_epub_path_ridi_2nd = os.path.join(epub_dir, epub_filename_ridi_2nd)
             counter_ridi = 1
             while os.path.exists(mod_epub_path_ridi_2nd):
                 mod_epub_path_ridi_2nd = os.path.join(epub_dir, f"{epub_filename_base_2nd}{epub_suffix_ridi_2nd.replace('.epub', '')} ({counter_ridi}).epub")
                 counter_ridi += 1

             returned_path_ridi_2nd = rebuild_epub_orchestrator(
                 epub_path=epub_file_path,
                 json_data=json_data_2nd,
                 updated_opf_soup=copy.deepcopy(updated_opf_soup),
                 updated_metadata=copy.deepcopy(updated_metadata),
                 model=model,
                 cover_image_modify=cover_image_modify,
                 cover_text_position=cover_text_position,
                 cover_text=cover_text,
                 font_path=font_path,
                 font_size=font_size,
                 font_color=font_color,
                 background_color=background_color,
                 translated_toc_map=final_toc_files_map_for_ridi_2nd,
                 output_epub_path=mod_epub_path_ridi_2nd,
                 mode='ridi'
             )

             if returned_path_ridi_2nd:
                 print_colored("2차 RIDI 버전 EPUB 생성 성공!", colorama.Fore.GREEN, colorama.Style.BRIGHT)
             else:
                 print_colored("Error: 2차 RIDI 버전 EPUB 파일 생성에 실패했습니다.", colorama.Fore.RED, colorama.Style.BRIGHT)
         else:
            pass
    else:
         print_colored("Error: 2차 표준 번역 EPUB 파일 생성에 실패하여 RIDI 버전 생성을 건너뜁니다.", colorama.Fore.RED, colorama.Style.BRIGHT)


    final_total_input_chars = total_input_chars_1st + total_input_chars_2nd
    final_total_output_chars = total_output_chars_1st + total_output_chars_2nd

    end_time = time.time()
    elapsed_time = end_time - start_time
    estimated_total_tokens = final_total_input_chars * 0.6 + final_total_output_chars * 0.75
    # Need to import datetime module for timedelta
    import datetime

    print(f"\n--- 최종 번역 결과 (1차 + 2차) ---")
    print(f"총 번역 시간: {str(datetime.timedelta(seconds=int(elapsed_time)))}")
    print(f"  - 1차 번역 Input: {int(total_input_chars_1st)} 글자, Output: {int(total_output_chars_1st)} 글자")
    print(f"  - 2차 번역 Input: {int(total_input_chars_2nd)} 글자, Output: {int(total_output_chars_2nd)} 글자")
    print(f"총 예상 사용 토큰 수: {int(estimated_total_tokens)} 토큰")
    if estimated_total_cost > 0:
        print(f"총 예상 비용: 약 ${estimated_total_cost:.4f}")

    return json_data_2nd, json_2nd_filename, returned_path_standard_2nd
    

def translate_single_line_2nd(filepath, model, prompt_details_2nd, line_pbar, api_call_delay):
    """
    2차 번역을 위해 단일 *한국어* 라인 파일을 처리하고 스트리밍 진행률을 업데이트합니다.
    """
    original_text = ""
    input_chars = 0
    output_chars = 0
    processed_text = ""
    status = 'failure_line_2nd' # 기본 상태: 실패
    pbar_non_whitespace_updated_line_2nd = 0

    try:
        # 1. 한국어 라인 파일 읽기
        with open(filepath, 'r', encoding='utf-8') as f:
            original_text = f.read()
        processed_text = original_text # 기본값: 원본
        original_non_whitespace_line_2nd = count_non_whitespace(original_text)

        # 2. 2차 번역 프롬프트 생성 (prev_context 없음)
        prompt = create_prompt_2nd(
            prompt_details_2nd["additional_instructions"],
            prompt_details_2nd["glossary"],
            prompt_details_2nd["characters"],
            {"prev_context": "", "current_text": original_text}
        )
        input_chars = len(prompt)

        # 3. API 호출 (재시도 및 스트리밍 포함, 3회 시도)
        retry_attempts_line = 3
        stream_successful = False
        for retry in range(retry_attempts_line):
            try:
                if api_call_delay > 0: time.sleep(api_call_delay)
                logging.info(f"Streaming 2nd Line Translation attempt {retry + 1} for {os.path.basename(filepath)}")

                response_stream = model.generate_content(prompt, stream=True)
                current_full_translated_text = ""
                current_line_output_chars = 0

                for chunk in response_stream:
                    chunk_text = ""
                    try:
                        if hasattr(chunk, 'parts') and chunk.parts: chunk_text = chunk.parts[0].text
                        elif hasattr(chunk, 'text'): chunk_text = chunk.text
                        else: continue
                    except Exception as chunk_err:
                         logging.error(f"Chunk error 2nd line trans {os.path.basename(filepath)}: {chunk_err}"); continue

                    current_full_translated_text += chunk_text
                    current_line_output_chars += len(chunk_text)

                    # 진행률 업데이트
                    non_whitespace_chunk_chars = count_non_whitespace(chunk_text)
                    if line_pbar and non_whitespace_chunk_chars > 0:
                        line_pbar.update(non_whitespace_chunk_chars)
                        pbar_non_whitespace_updated_line_2nd += non_whitespace_chunk_chars

                if not current_full_translated_text.strip():
                     logging.warning(f"Empty result stream 2nd line trans {os.path.basename(filepath)} (attempt {retry+1}).")
                     if retry < retry_attempts_line - 1: time.sleep(5)
                     continue

                # 성공 시 후처리 및 결과 저장
                processed_text_attempt = apply_regex_transformations(current_full_translated_text) # 기본 클리닝 적용
                processed_text = processed_text_attempt
                output_chars = len(processed_text)
                status = 'success_line_2nd' # 성공 상태로 변경
                stream_successful = True
                logging.info(f"Streaming 2nd line trans success for {os.path.basename(filepath)} (attempt {retry+1})")
                break # 성공 시 재시도 중단

            except InvalidArgument as iae:
                 logging.error(f"InvalidArgument API Error during 2nd line translation for {filepath} (retry {retry+1}): {iae}. Keeping original.")
                 processed_text = original_text; output_chars = len(original_text); status = 'failure_line_2nd_invalid_arg'
                 break # 재시도 중단
            except Exception as e:
                logging.warning(f"API Error during 2nd line translation for {filepath} (retry {retry+1}/{retry_attempts_line}): {e}")
                if retry < retry_attempts_line - 1: time.sleep(10)
                else: # 최종 실패
                     processed_text = original_text; output_chars = len(original_text); status = 'failure_line_2nd_api'

        # 최종 진행률 보정
        if line_pbar:
            correction = original_non_whitespace_line_2nd - pbar_non_whitespace_updated_line_2nd
            if correction != 0:
                logging.debug(f"Applying final progress correction 2nd line trans {os.path.basename(filepath)} (Status: {status}): {correction}")
                line_pbar.update(correction)

        # 파일에 최종 결과 쓰기
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(processed_text)

    except Exception as line_proc_e:
         logging.error(f"Error processing file {filepath} in translate_single_line_2nd: {line_proc_e}", exc_info=True)
         processed_text = original_text if original_text else "" # 원본 유지 또는 빈 문자열
         output_chars = len(processed_text)
         status = 'failure_line_2nd_processing'

    # filepath와 함께 결과 반환 (호출 측에서 식별하기 위함)
    return filepath, status, input_chars, output_chars


def translate_lines_2nd(output_dir, line_files, model, prompt_details_2nd, pbar_fallback, api_call_delay, num_parallel, total_lines_in_fallback, completed_lines_ref):
    """
    2차 번역 실패 블록의 한국어 라인 파일들을 병렬로 처리합니다.
    """
    total_input_chars_lines = 0
    total_output_chars_lines = 0
    all_lines_successful = True # 해당 블록의 모든 줄이 성공했는지 추적

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_parallel) as executor:
        futures = [executor.submit(translate_single_line_2nd,
                                    filepath,
                                    model,
                                    prompt_details_2nd,
                                    pbar_fallback, # fallback용 pbar 전달
                                    api_call_delay
                                   ) for filepath in line_files]

        for future in concurrent.futures.as_completed(futures):
            try:
                fpath, line_status, input_c, output_c = future.result()
                total_input_chars_lines += input_c
                total_output_chars_lines += output_c

                if line_status != 'success_line_2nd':
                     all_lines_successful = False
                     logging.warning(f"2nd translation fallback: Line failed - {os.path.basename(fpath)} (Status: {line_status})")

                # 완료된 줄 수 업데이트 및 진행률 표시
                completed_lines_ref[0] += 1
                if pbar_fallback:
                    pbar_fallback.set_postfix({"완료 줄 수 (Fallback)": f"{completed_lines_ref[0]}/{total_lines_in_fallback}"}, refresh=True)

            except Exception as e:
                logging.error(f"Error processing future in translate_lines_2nd: {e}", exc_info=True)
                all_lines_successful = False
                # 에러 발생 시에도 카운트는 증가시켜야 함
                completed_lines_ref[0] += 1
                if pbar_fallback:
                     pbar_fallback.set_postfix({"완료 줄 수 (Fallback)": f"{completed_lines_ref[0]}/{total_lines_in_fallback}"}, refresh=True)

    return total_input_chars_lines, total_output_chars_lines, all_lines_successful
    

def split_text_block_2nd(filepath):
    """
    2차 번역 대상 텍스트 블록 파일(_2nd.txt)을 줄 단위로 분할하고,
    순서 정보를 담은 JSON 파일을 생성합니다. (빈 줄 제외) - 2차 번역 fallback용
    """
    output_dir = os.path.dirname(filepath)
    # 입력 파일 이름에서 '_2nd.txt' 제거하고 기본 이름 추출
    base_filename_match = re.match(r"(text_block_\d+)_2nd\.txt$", os.path.basename(filepath))
    if not base_filename_match:
        logging.error(f"Invalid filename format for 2nd split: {filepath}")
        return None, [] # 잘못된 파일명 형식 처리
    base_filename = base_filename_match.group(1) # 예: "text_block_1"

    # 순서 정보 JSON 파일 경로 (구별되는 이름 사용)
    line_order_json_filepath = os.path.join(output_dir, f"{base_filename}_2nd_order.json")
    line_files = []
    line_data = []

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        line_count = 0
        for i, line in enumerate(lines):
            line_content = line.strip(' \t\n\r') # 앞뒤 공백/개행만 제거
            if not line_content:
                continue

            line_count += 1
            # 라인 파일 이름 형식 변경 (예: text_block_1_2nd_1.txt)
            line_filename = os.path.join(output_dir, f"{base_filename}_2nd_{line_count}.txt")
            with open(line_filename, 'w', encoding='utf-8') as line_file:
                line_file.write(line_content) # strip된 내용 저장

            line_data.append({
                "order": line_count,
                "content": line_content # 저장된 내용과 동일하게 JSON에도 저장
            })
            line_files.append(line_filename)

        # 순서 정보 JSON 저장
        with open(line_order_json_filepath, 'w', encoding='utf-8') as json_file:
            json.dump(line_data, json_file, indent=4, ensure_ascii=False)

        return line_order_json_filepath, line_files

    except Exception as e:
        logging.error(f"Error splitting 2nd text block {filepath}: {e}", exc_info=True)
        # 오류 발생 시 생성된 파일들 정리 (선택적)
        for lf in line_files:
             if os.path.exists(lf): os.remove(lf)
        if os.path.exists(line_order_json_filepath): os.remove(line_order_json_filepath)
        return None, []

        
def identify_and_extract_toc_files(epub_path, output_dir):
    ncx_info = None
    nav_doc_info = None

    try:
        with zipfile.ZipFile(epub_path, 'r') as zin:
            opf_path_internal = find_opf_file(zin) # Assume find_opf_file is defined
            if not opf_path_internal:
                logging.error("OPF 파일을 찾을 수 없습니다.")
                return None, None

            opf_path = posixpath.normpath(opf_path_internal)
            opf_dir = posixpath.dirname(opf_path) if opf_path else ''

            opf_content_bytes = zin.read(opf_path)
            if opf_content_bytes.startswith(b'\xef\xbb\xbf'):
                opf_content_bytes = opf_content_bytes[3:]

            opf_content = None
            try:
                 opf_content = opf_content_bytes.decode('utf-8')
            except UnicodeDecodeError:
                 try:
                      opf_content = opf_content_bytes.decode('cp949')
                      logging.info("OPF 파일을 cp949로 디코딩했습니다.")
                 except Exception as dec_err:
                      logging.error(f"OPF 파일 디코딩 실패: {dec_err}. 목차 파일 식별 불가.")
                      return None, None

            if not opf_content: return None, None

            opf_soup = BeautifulSoup(opf_content, 'xml')
            manifest = opf_soup.find('manifest')
            if not manifest:
                logging.error("OPF 파일에서 <manifest>를 찾을 수 없습니다.")
                return None, None

            # Nav Doc 찾기 (properties="nav")
            nav_item = manifest.find('item', attrs={'properties': re.compile(r'\bnav\b')})
            if nav_item and nav_item.get('href'):
                nav_href = nav_item['href']
                nav_epub_path = posixpath.normpath(posixpath.join(opf_dir, nav_href))
                original_nav_filename = posixpath.basename(nav_epub_path)
                temp_nav_filename = f"original_{original_nav_filename}"
                temp_nav_filepath = os.path.join(output_dir, temp_nav_filename)
                try:
                    # Ensure parent directory exists before writing
                    os.makedirs(os.path.dirname(temp_nav_filepath), exist_ok=True)
                    with zin.open(nav_epub_path) as source, open(temp_nav_filepath, 'wb') as target:
                        shutil.copyfileobj(source, target)
                    nav_doc_info = {'epub_path': nav_epub_path, 'temp_path': temp_nav_filepath, 'original_filename': original_nav_filename}
                    logging.info(f"Nav Doc 원본 저장 완료: {temp_nav_filepath}")
                except KeyError:
                     logging.error(f"EPUB 아카이브에서 Nav Doc 파일({nav_epub_path})을 찾을 수 없습니다.")
                except Exception as copy_err:
                     logging.error(f"Nav Doc 파일 복사 실패 ({nav_epub_path} -> {temp_nav_filepath}): {copy_err}")

            # NCX 찾기 (id="ncx" 또는 spine toc 속성)
            ncx_item = manifest.find('item', attrs={'id': 'ncx'})
            if not ncx_item:
                 spine = opf_soup.find('spine')
                 if spine and spine.get('toc'):
                      ncx_id_from_spine = spine['toc']
                      ncx_item = manifest.find('item', attrs={'id': ncx_id_from_spine})

            if ncx_item and ncx_item.get('href'):
                ncx_href = ncx_item['href']
                ncx_epub_path = posixpath.normpath(posixpath.join(opf_dir, ncx_href))
                original_ncx_filename = posixpath.basename(ncx_epub_path)
                temp_ncx_filename = f"original_{original_ncx_filename}"
                temp_ncx_filepath = os.path.join(output_dir, temp_ncx_filename)
                try:
                    # Ensure parent directory exists before writing
                    os.makedirs(os.path.dirname(temp_ncx_filepath), exist_ok=True)
                    with zin.open(ncx_epub_path) as source, open(temp_ncx_filepath, 'wb') as target:
                         shutil.copyfileobj(source, target)
                    ncx_info = {'epub_path': ncx_epub_path, 'temp_path': temp_ncx_filepath, 'original_filename': original_ncx_filename}
                    logging.info(f"NCX 원본 저장 완료: {temp_ncx_filepath}")
                except KeyError:
                     logging.error(f"EPUB 아카이브에서 NCX 파일({ncx_epub_path})을 찾을 수 없습니다.")
                except Exception as copy_err:
                     logging.error(f"NCX 파일 복사 실패 ({ncx_epub_path} -> {temp_ncx_filepath}): {copy_err}")

            if not nav_doc_info and not ncx_info:
                 logging.warning("OPF에서 EPUB3 Nav Doc 또는 EPUB2 NCX 파일을 찾지 못했습니다.")

            return ncx_info, nav_doc_info

    except FileNotFoundError as e:
        print_colored(f"오류: 목차 파일 식별 중 파일 오류 - {e}", colorama.Fore.RED)
        logging.error(f"FileNotFoundError during TOC identification: {e}", exc_info=True)
        return None, None
    except Exception as e:
        print_colored(f"오류: 목차 파일 식별/추출 중 예외 발생: {e}", colorama.Fore.RED)
        logging.error(f"Exception during TOC identification/extraction: {e}", exc_info=True)
        return None, None
        

def translate_ncx_file(ncx_temp_path, output_dir, model, updated_metadata, api_call_delay, original_filename):
    logging.info(f"NCX 파일 번역 시작: {ncx_temp_path} (원본명: {original_filename})")
    last_api_call_time_ref = [0.0]
    processed_ncx_bytes = None
    ncx_original_bytes = None # Read original bytes for comparison

    try:
        with open(ncx_temp_path, 'rb') as f_in:
            ncx_original_bytes = f_in.read()

        # Assume process_ncx function is defined elsewhere and uses global NAV_TRANSLATIONS
        processed_ncx_bytes = process_ncx(ncx_original_bytes, updated_metadata, NAV_TRANSLATIONS, model, api_call_delay, last_api_call_time_ref)

        # Check if processing returned valid bytes and if it's different from original
        if processed_ncx_bytes is not None and processed_ncx_bytes != ncx_original_bytes:
            translated_filepath = os.path.join(output_dir, original_filename)
            try:
                os.makedirs(os.path.dirname(translated_filepath), exist_ok=True)
                with open(translated_filepath, 'wb') as f_out:
                    f_out.write(processed_ncx_bytes)
                logging.info(f"번역된 NCX 파일 저장 완료 (파일명 유지): {translated_filepath}")
                return translated_filepath
            except Exception as write_err:
                logging.error(f"번역된 NCX 파일 저장 실패 ({translated_filepath}): {write_err}")
                return None
        elif processed_ncx_bytes is not None: # Processed, but no changes
             logging.info(f"NCX 파일 처리되었으나 변경 사항 없음: {original_filename}")
             return None # Indicate no translated version to use (main will use original temp)
        else: # Processing failed
             logging.warning(f"NCX 파일 처리 실패: {ncx_temp_path}")
             return None

    except FileNotFoundError:
         logging.error(f"NCX 임시 파일을 찾을 수 없음: {ncx_temp_path}")
         return None
    except Exception as e:
         logging.error(f"NCX 번역 중 오류 발생 ({ncx_temp_path}): {e}", exc_info=True)
         return None


def translate_nav_doc_file(nav_doc_temp_path, output_dir, model, updated_metadata, api_call_delay, original_filename):
    logging.info(f"Nav Doc 파일 번역 시작: {nav_doc_temp_path} (원본명: {original_filename})")
    last_api_call_time_ref = [0.0]
    processed_nav_bytes = None
    nav_original_bytes = None # Read original bytes for comparison

    try:
        with open(nav_doc_temp_path, 'rb') as f_in:
            nav_original_bytes = f_in.read()

        # Assume process_nav_doc function is defined elsewhere and uses global NAV_TRANSLATIONS
        # Pass mode='standard' as RIDI headers are handled later
        processed_nav_bytes = process_nav_doc(nav_original_bytes, updated_metadata, model, api_call_delay, last_api_call_time_ref, mode='standard')

        # Check if processing returned valid bytes and if it's different from original
        if processed_nav_bytes is not None and processed_nav_bytes != nav_original_bytes:
            translated_filepath = os.path.join(output_dir, original_filename)
            try:
                os.makedirs(os.path.dirname(translated_filepath), exist_ok=True)
                with open(translated_filepath, 'wb') as f_out:
                    f_out.write(processed_nav_bytes)
                logging.info(f"번역된 Nav Doc 파일 저장 완료 (파일명 유지): {translated_filepath}")
                return translated_filepath
            except Exception as write_err:
                logging.error(f"번역된 Nav Doc 파일 저장 실패 ({translated_filepath}): {write_err}")
                return None
        elif processed_nav_bytes is not None: # Processed, but no changes
             logging.info(f"Nav Doc 파일 처리되었으나 변경 사항 없음: {original_filename}")
             return None # Indicate no translated version to use
        else: # Processing failed
             logging.warning(f"Nav Doc 파일 처리 실패: {nav_doc_temp_path}")
             return None

    except FileNotFoundError:
         logging.error(f"Nav Doc 임시 파일을 찾을 수 없음: {nav_doc_temp_path}")
         return None
    except Exception as e:
         logging.error(f"Nav Doc 번역 중 오류 발생 ({nav_doc_temp_path}): {e}", exc_info=True)
         return None


def convert_nav_html_to_ncx(nav_html_content, updated_metadata):
    """
    처리된 Nav Doc HTML 내용을 기반으로 toc.ncx 파일 내용을 생성합니다.

    Args:
        nav_html_content (str): 번역 및 처리된 Nav Doc HTML 문자열.
        updated_metadata (dict): 'title' 등을 포함한 업데이트된 메타데이터 (구조화된 형태).

    Returns:
        bytes: 생성된 NCX 파일 내용 (UTF-8 인코딩), 실패 시 None.
    """
    try:
        logging.info("Nav Doc HTML을 NCX 형식으로 변환 시작...")
        soup = BeautifulSoup(nav_html_content, 'html.parser')

        ncx_namespace = "http://www.daisy.org/z3986/2005/ncx/"
        ET.register_namespace('', ncx_namespace)
        ncx_root = ET.Element(f"{{{ncx_namespace}}}ncx", version="2005-1", attrib={"{http://www.w3.org/XML/1998/namespace}lang": "ko"})

        head = ET.SubElement(ncx_root, f"{{{ncx_namespace}}}head")
        uid_str = f"urn:uuid:{uuid.uuid4()}"
        ET.SubElement(head, f"{{{ncx_namespace}}}meta", name="dtb:uid", content=uid_str)
        depth_meta = ET.SubElement(head, f"{{{ncx_namespace}}}meta", name="dtb:depth", content="1")
        ET.SubElement(head, f"{{{ncx_namespace}}}meta", name="dtb:totalPageCount", content="0")
        ET.SubElement(head, f"{{{ncx_namespace}}}meta", name="dtb:maxPageNumber", content="0")

        doc_title = ET.SubElement(ncx_root, f"{{{ncx_namespace}}}docTitle")
        doc_title_text = ET.SubElement(doc_title, f"{{{ncx_namespace}}}text")
        # Use the 'value' from the updated_metadata dictionary for title
        title_info = updated_metadata.get('title')
        doc_title_string = title_info.get('value', '제목 없음') if isinstance(title_info, dict) else '제목 없음'
        doc_title_text.text = doc_title_string

        nav_map = ET.SubElement(ncx_root, f"{{{ncx_namespace}}}navMap")

        play_order_counter = 1
        max_depth = 0

        def process_html_list(html_ol_or_ul, parent_navpoint, current_depth):
            nonlocal play_order_counter, max_depth
            max_depth = max(max_depth, current_depth)

            for li in html_ol_or_ul.find_all('li', recursive=False):
                a_tag = li.find('a', recursive=False)
                if a_tag and a_tag.get('href'):
                    nav_point = ET.SubElement(parent_navpoint, f"{{{ncx_namespace}}}navPoint",
                                              id=f"navpoint-{play_order_counter}",
                                              playOrder=str(play_order_counter))
                    play_order_counter += 1

                    nav_label = ET.SubElement(nav_point, f"{{{ncx_namespace}}}navLabel")
                    text_elem = ET.SubElement(nav_label, f"{{{ncx_namespace}}}text")

                    # ☆☆☆ 수정된 부분: label_text 타입 확인 및 처리 ☆☆☆
                    label_text_raw = a_tag.get_text(strip=True)
                    # 명시적으로 문자열로 변환하거나, 타입 체크 후 오류 처리
                    if isinstance(label_text_raw, str):
                        label_text = label_text_raw
                    else:
                        # 비 문자열 값 발견 시 로깅 및 대체 텍스트 사용
                        logging.error(f"NCX 변환 중 <a> 태그에서 예기치 않은 비 문자열 내용 발견: {label_text_raw} (타입: {type(label_text_raw)}). 대체 텍스트 사용.")
                        label_text = "[내용 오류]" # 또는 다른 적절한 대체 값

                    text_elem.text = label_text if label_text else " " # 빈 문자열 방지
                    # ☆☆☆ 수정된 부분 끝 ☆☆☆

                    content_elem = ET.SubElement(nav_point, f"{{{ncx_namespace}}}content",
                                                 src=a_tag['href'])

                    nested_list = li.find(['ol', 'ul'], recursive=False)
                    if nested_list:
                        process_html_list(nested_list, nav_point, current_depth + 1)

        toc_nav = soup.find('nav', attrs={'epub:type': re.compile(r'\btoc\b')})
        if not toc_nav: toc_nav = soup.find('nav')
        if not toc_nav: toc_nav = soup.body

        if toc_nav:
            top_level_list = toc_nav.find(['ol', 'ul'], recursive=False)
            if top_level_list:
                process_html_list(top_level_list, nav_map, 1)
            else:
                logging.warning("NCX 변환: Nav Doc에서 최상위 목록(ol/ul)을 찾지 못했습니다.")
        else:
            logging.warning("NCX 변환: Nav Doc에서 <nav> 또는 body 요소를 찾지 못했습니다.")

        depth_meta.set('content', str(max_depth))

        # XML 직렬화 (minidom 사용)
        # ElementTree -> bytes -> string -> minidom -> bytes (조금 비효율적일 수 있으나 pretty print 위해)
        raw_string_bytes = ET.tostring(ncx_root, encoding='utf-8', method='xml', xml_declaration=True)
        parsed_xml = minidom.parseString(raw_string_bytes)
        pretty_xml_lines = parsed_xml.toprettyxml(indent="  ", encoding='utf-8').splitlines()

        pretty_xml_no_blank_lines = [line for line in pretty_xml_lines if line.strip()]
        final_ncx_bytes = b"\n".join(pretty_xml_no_blank_lines)

        logging.info(f"NCX 변환 완료. (UID: {uid_str}, Depth: {max_depth})")
        return final_ncx_bytes

    except Exception as e:
        # ☆☆☆ 예외 발생 시 로깅 강화 ☆☆☆
        logging.error(f"Nav Doc을 NCX로 변환하는 중 오류 발생: {e}", exc_info=True)
        # 오류의 원인이 된 HTML 내용 일부 로깅 (디버깅 목적, 민감 정보 주의)
        try:
            faulty_html_snippet = nav_html_content[max(0, e.lineno-5):e.lineno+5] if hasattr(e, 'lineno') else nav_html_content[:500] # 대략적인 위치
            logging.error(f"오류 발생 지점 근처 HTML 내용 (일부): \n{faulty_html_snippet}")
        except: pass # 로깅 중 추가 오류 방지
        return None

        
# --- 5. EPUB 재구성 ---
        
def get_and_update_metadata(epub_path):
    """EPUB 파일에서 메타데이터를 읽고 사용자 입력을 받아 업데이트합니다."""
    try:
        with zipfile.ZipFile(epub_path, 'r') as epub_file:
            opf_path = find_opf_file(epub_file)
            if not opf_path:
                raise FileNotFoundError("OPF 파일을 찾을 수 없습니다.")

            with epub_file.open(opf_path, 'r') as opf_content_file:
                opf_content = opf_content_file.read()
                if opf_content.startswith(b'\xef\xbb\xbf'):
                    opf_content = opf_content[3:]
                # ☆☆☆ 파서 명시 추가 ☆☆☆
                opf_soup = BeautifulSoup(opf_content, 'xml')

                metadata = opf_soup.find('metadata')
                if not metadata:
                    print_colored("Error: OPF 파일에서 <metadata> 태그를 찾을 수 없습니다.", colorama.Fore.RED, colorama.Style.BRIGHT)
                    return None, None

                # ☆☆☆ updated_metadata 구조 변경 ☆☆☆
                updated_metadata = {
                    'title': None,      # {'id': 'title_id', 'value': 'new_title'}
                    'creators': [],     # [{'id': 'creator_id', 'value': 'new_creator'}, ...]
                    'publisher': None   # {'id': 'pub_id', 'value': 'new_publisher'}
                }

                print("\n--- EPUB 메타데이터 수정 ---")

                # --- 제목(Title) 처리 ---
                title_tag = metadata.find('dc:title')
                if title_tag:
                    title_id = title_tag.get('id')
                    existing_value = title_tag.string.strip() if title_tag.string else ""
                    new_value_input = input(f"제목 입력 (ID: {title_id or '없음'}, 기존: '{existing_value}', Enter시 유지): ").strip()
                    final_value = new_value_input if new_value_input else existing_value
                    if final_value: # 값이 있을 때만 저장
                        updated_metadata['title'] = {'id': title_id, 'value': final_value}
                else:
                    print_colored("경고: <dc:title> 태그를 찾을 수 없습니다.", colorama.Fore.YELLOW)

                # --- 저자(Creators) 처리 (모든 저자) ---
                creator_tags = metadata.find_all('dc:creator')
                if creator_tags:
                    for i, creator_tag in enumerate(creator_tags):
                        creator_id = creator_tag.get('id')
                        existing_value = creator_tag.string.strip() if creator_tag.string else ""
                        # ID가 없는 경우 임시 ID 부여 (수정 시 필요)
                        if not creator_id:
                            creator_id = f"temp_creator_{i+1}"
                            creator_tag['id'] = creator_id # soup 객체에 임시 ID 추가 (나중에 refines에 사용)
                            #print_colored(f"경고: ID가 없는 저자 태그에 임시 ID '{creator_id}' 부여.", colorama.Fore.YELLOW)

                        prompt_text = f"저자{i+1} 입력 (ID: {creator_id}, 기존: '{existing_value}', Enter시 유지): "
                        new_value_input = input(prompt_text).strip()
                        final_value = new_value_input if new_value_input else existing_value
                        if final_value: # 값이 있을 때만 저장
                            updated_metadata['creators'].append({'id': creator_id, 'value': final_value})
                else:
                    print_colored("경고: <dc:creator> 태그를 찾을 수 없습니다.", colorama.Fore.YELLOW)

                # --- 출판사(Publisher) 처리 ---
                publisher_tag = metadata.find('dc:publisher')
                if publisher_tag:
                    publisher_id = publisher_tag.get('id')
                    existing_value = publisher_tag.string.strip() if publisher_tag.string else ""
                    new_value_input = input(f"출판사 입력 (ID: {publisher_id or '없음'}, 기존: '{existing_value}', Enter시 유지): ").strip()
                    final_value = new_value_input if new_value_input else existing_value
                    if final_value: # 값이 있을 때만 저장
                        updated_metadata['publisher'] = {'id': publisher_id, 'value': final_value}
                else:
                    print_colored("경고: <dc:publisher> 태그를 찾을 수 없습니다.", colorama.Fore.YELLOW)

                # --- 기타 필요한 메타데이터 처리 로직 추가 가능 ---

                print("---------------------------\n")
                # ☆☆☆ opf_soup 객체와 구조화된 updated_metadata 반환 ☆☆☆
                return opf_soup, updated_metadata

    except FileNotFoundError as e:
        print_colored(f"Error: 메타데이터 처리 중 오류 - {e}", colorama.Fore.RED, colorama.Style.BRIGHT)
        return None, None
    except Exception as e:
        print_colored(f"Error: 메타데이터 읽기/수정 중 예외 발생: {e}", colorama.Fore.RED, colorama.Style.BRIGHT)
        import traceback
        traceback.print_exc()
        return None, None


def read_original_epub(epub_path):
    """
    원본 EPUB 파일을 열고 모든 파일의 내용과 메타데이터(ZipInfo)를 로드합니다.
    OPF 파일 경로를 찾아 반환합니다.

    Args:
        epub_path (str): 원본 EPUB 파일 경로.

    Returns:
        tuple: (original_content_map, original_zipinfo_map, opf_path, opf_dir)
               original_content_map (dict): {파일 경로: 파일 내용(bytes)}
               original_zipinfo_map (dict): {파일 경로: ZipInfo 객체}
               opf_path (str or None): OPF 파일 경로 (EPUB 내부 기준).
               opf_dir (str): OPF 파일이 있는 디렉토리 경로.
               None 반환 시 오류 발생.
    """
    original_content_map = {}
    original_zipinfo_map = {}
    opf_path = None
    opf_dir = ''

    try:
        with zipfile.ZipFile(epub_path, 'r') as zin:
            opf_path_internal = find_opf_file(zin) # find_opf_file 은 이미 정의되어 있다고 가정
            if not opf_path_internal:
                raise FileNotFoundError("원본 EPUB에서 OPF 파일을 찾을 수 없습니다.")

            opf_path = posixpath.normpath(opf_path_internal)
            opf_dir = posixpath.dirname(opf_path) if opf_path else ''

            logging.info("원본 EPUB 파일 내용 읽는 중...")
            for item in zin.infolist():
                item_path_norm = posixpath.normpath(item.filename)
                # 디렉토리는 내용이 없으므로 빈 바이트 저장 또는 건너뛰기 가능
                if item.is_dir():
                     original_content_map[item_path_norm] = b'' # 또는 continue
                else:
                     original_content_map[item_path_norm] = zin.read(item.filename)
                original_zipinfo_map[item_path_norm] = item
            logging.info(f"원본 EPUB에서 {len(original_content_map)}개의 항목(파일/디렉토리)을 로드했습니다.")

        return original_content_map, original_zipinfo_map, opf_path, opf_dir

    except FileNotFoundError as e:
        print_colored(f"오류: EPUB 읽기 중 파일 오류 - {e}", colorama.Fore.RED)
        logging.error(f"FileNotFoundError during EPUB read: {e}", exc_info=True)
        return None, None, None, None
    except zipfile.BadZipFile:
        print_colored(f"오류: '{epub_path}'는 유효한 EPUB/ZIP 파일이 아닙니다.", colorama.Fore.RED)
        logging.error(f"BadZipFile error reading: {epub_path}")
        return None, None, None, None
    except Exception as e:
        print_colored(f"오류: 원본 EPUB 읽기 중 예외 발생: {e}", colorama.Fore.RED)
        logging.error(f"Unexpected error reading original EPUB: {e}", exc_info=True)
        return None, None, None, None
        

def identify_linked_css(original_content_map, original_zipinfo_map, json_data, opf_dir):
    """
    json_data에 포함된 번역 대상 XHTML 파일들을 분석하여,
    해당 파일들에서 링크된 유효한 CSS 파일들의 절대 경로(EPUB 내부 기준) 목록을 식별합니다.

    Args:
        original_content_map (dict): 원본 파일 내용 맵.
        original_zipinfo_map (dict): 원본 ZipInfo 맵.
        json_data (dict): EPUB 구조 정보 JSON 데이터.
        opf_dir (str): OPF 파일 기준 디렉토리 경로.

    Returns:
        set: 링크된 CSS 파일 경로들의 집합.
    """
    linked_css_paths = set()
    json_xhtml_keys = {k for k in json_data if k != "epub_filename"}
    xhtml_path_map = {} # json 키(파일명) -> EPUB 내 실제 경로 매핑

    # json_data 키와 실제 EPUB 내 파일 경로 매핑
    for item_path_norm, item_info in original_zipinfo_map.items():
        if not item_info.is_dir() and item_path_norm.lower().endswith((".xhtml", ".html")):
            base_name = posixpath.basename(item_path_norm)
            if base_name in json_xhtml_keys:
                xhtml_path_map[base_name] = item_path_norm

    if not xhtml_path_map:
        logging.warning("JSON 데이터에 명시된 XHTML 파일들을 EPUB 아카이브에서 찾을 수 없습니다.")
        return linked_css_paths

    logging.info("번역 대상 XHTML 파일들에서 링크된 CSS 파일 찾는 중...")
    for xhtml_base_name, xhtml_full_path in xhtml_path_map.items():
        xhtml_content_bytes = original_content_map.get(xhtml_full_path)
        if not xhtml_content_bytes:
            logging.warning(f"XHTML 파일 내용을 찾을 수 없습니다: {xhtml_full_path}. CSS 링크 스캔 건너뜁니다.")
            continue

        try:
            # XHTML 디코딩 (BOM 처리 포함)
            detected_encoding = 'utf-8'
            if xhtml_content_bytes.startswith(b'\xef\xbb\xbf'):
                xhtml_content_bytes_no_bom = xhtml_content_bytes[3:]
            else:
                xhtml_content_bytes_no_bom = xhtml_content_bytes

            xhtml_content = None
            try:
                xhtml_content = xhtml_content_bytes_no_bom.decode(detected_encoding)
            except UnicodeDecodeError:
                try:
                    detected_encoding = 'cp949' # 또는 다른 예상 인코딩
                    xhtml_content = xhtml_content_bytes_no_bom.decode(detected_encoding)
                    logging.info(f"{xhtml_full_path} 파일을 {detected_encoding}으로 디코딩했습니다.")
                except Exception as decode_err_inner:
                    logging.warning(f"XHTML 파일 디코딩 실패 ({xhtml_full_path}): {decode_err_inner}. CSS 링크 스캔 건너뜁니다.")
                    continue
            except Exception as decode_err_outer:
                 logging.warning(f"XHTML 파일 디코딩 중 오류 ({xhtml_full_path}): {decode_err_outer}. CSS 링크 스캔 건너뜁니다.")
                 continue

            if xhtml_content is None: continue # 디코딩 최종 실패 시 건너뜀

            # BeautifulSoup으로 파싱하여 CSS 링크 찾기
            soup_xhtml = BeautifulSoup(xhtml_content, 'html.parser')
            head = soup_xhtml.find('head')
            if head:
                xhtml_dir = posixpath.dirname(xhtml_full_path)
                for link_tag in head.find_all('link', rel='stylesheet'):
                    href = link_tag.get('href')
                    if href:
                        href_cleaned = href.split('#')[0] # Fragment 제거
                        # CSS 경로 해석 (절대 경로 또는 상대 경로)
                        if posixpath.isabs(href_cleaned):
                             css_abs_path = posixpath.normpath(href_cleaned.lstrip('/')) # 루트 기준 절대 경로
                        else:
                             css_abs_path = posixpath.normpath(posixpath.join(xhtml_dir, href_cleaned))

                        # CSS 파일이 실제로 존재하고 디렉토리가 아닌지 확인
                        if css_abs_path in original_zipinfo_map and not original_zipinfo_map[css_abs_path].is_dir():
                            linked_css_paths.add(css_abs_path)
                            logging.debug(f"    - 발견된 CSS 링크: {css_abs_path} (From: {xhtml_full_path})")
                        else:
                            logging.warning(f"링크된 CSS 파일을 찾을 수 없거나 디렉토리입니다: '{css_abs_path}' (In: {xhtml_full_path})")
        except Exception as e_parse:
            logging.warning(f"XHTML 파싱 중 오류 발생 ({xhtml_full_path}): {e_parse}. CSS 링크 스캔 건너뜁니다.")

    logging.info(f"총 {len(linked_css_paths)}개의 고유하게 링크된 CSS 파일을 식별했습니다.")
    return linked_css_paths

def process_all_css(original_content_map, original_zipinfo_map, linked_css_paths, korean_style_content):
    """
    모든 CSS 파일을 순회하며 내용을 수정합니다.
    (vertical writing-mode 제거, 링크된 CSS에는 korean_style 추가)

    Args:
        original_content_map (dict): 원본 파일 내용 맵.
        original_zipinfo_map (dict): 원본 ZipInfo 맵.
        linked_css_paths (set): 링크된 CSS 파일 경로 집합.
        korean_style_content (str): 추가할 한국어 스타일 CSS 내용.

    Returns:
        dict: {CSS 파일 경로: 수정된 CSS 내용(bytes)}
    """
    modified_css_content = {}
    logging.info("모든 CSS 파일 처리 중...")

    for item_path, item_info in original_zipinfo_map.items():
        # CSS 파일만 대상으로 함
        if not item_info.is_dir() and item_path.lower().endswith(".css"):
            item_bytes = original_content_map.get(item_path)
            if item_bytes is None:
                logging.warning(f"CSS 파일 내용을 찾을 수 없습니다: {item_path}. 건너뜁니다.")
                continue

            try:
                css_content = None
                detected_encoding = 'utf-8'
                css_changed = False

                # BOM 제거
                if item_bytes.startswith(b'\xef\xbb\xbf'):
                    item_bytes_no_bom = item_bytes[3:]
                else:
                    item_bytes_no_bom = item_bytes

                # CSS 디코딩 시도 (@charset 고려)
                try:
                    css_content_test = item_bytes_no_bom.decode(detected_encoding)
                    # @charset 규칙 확인
                    charset_match = re.match(r'@charset\s+"([^"]+)"\s*;', css_content_test, re.IGNORECASE)
                    if charset_match:
                        encoding_in_css = charset_match.group(1).strip().lower()
                        # @charset에 명시된 인코딩이 utf-8이 아니면 해당 인코딩으로 다시 시도
                        if encoding_in_css and encoding_in_css != detected_encoding and encoding_in_css != 'utf8':
                            try:
                                css_content = item_bytes_no_bom.decode(encoding_in_css)
                                detected_encoding = encoding_in_css
                                logging.info(f"{item_path} 파일의 @charset({encoding_in_css})을 적용하여 디코딩했습니다.")
                            except Exception:
                                logging.warning(f"{item_path} 파일의 @charset({encoding_in_css})으로 디코딩 실패. UTF-8 사용.")
                                css_content = css_content_test # UTF-8 결과 사용
                        else:
                            css_content = css_content_test # UTF-8 결과 사용
                    else:
                        css_content = css_content_test # @charset 없으면 UTF-8 결과 사용
                except UnicodeDecodeError:
                    # UTF-8 실패 시 다른 인코딩 시도 (예: cp949)
                    try:
                        detected_encoding = 'cp949'
                        css_content = item_bytes_no_bom.decode(detected_encoding)
                        logging.info(f"{item_path} 파일을 {detected_encoding}으로 디코딩했습니다.")
                    except Exception as decode_err_inner:
                        logging.warning(f"CSS 파일 디코딩 최종 실패 ({item_path}): {decode_err_inner}. 원본 바이트 사용.")
                        modified_css_content[item_path] = item_bytes # 수정 불가, 원본 저장
                        continue
                except Exception as decode_err_outer:
                    logging.warning(f"CSS 파일 디코딩 중 오류 ({item_path}): {decode_err_outer}. 원본 바이트 사용.")
                    modified_css_content[item_path] = item_bytes
                    continue

                if css_content is None: # 디코딩 최종 실패 시
                    modified_css_content[item_path] = item_bytes
                    continue

                # vertical writing-mode 제거
                original_len = len(css_content)
                css_content = re.sub(r"writing-mode\s*:\s*(vertical-rl|vertical-lr)\s*(!important)?\s*;", "", css_content, flags=re.IGNORECASE)
                css_content = re.sub(r"-epub-writing-mode\s*:\s*(vertical-rl|vertical-lr)\s*(!important)?\s*;", "", css_content, flags=re.IGNORECASE)
                css_content = re.sub(r"-webkit-writing-mode\s*:\s*(vertical-rl|vertical-lr)\s*(!important)?\s*;", "", css_content, flags=re.IGNORECASE)
                if len(css_content) < original_len:
                    css_changed = True
                    logging.debug(f"    - {item_path}: Vertical writing-mode 속성을 제거했습니다.")

                # 링크된 CSS 파일에 korean_style 추가
                if item_path in linked_css_paths:
                    if "p.korean_style" not in css_content: # 중복 추가 방지
                        css_content += "\n\n/* Added by rebuild_epub */\n" + korean_style_content.strip() + "\n"
                        css_changed = True
                        logging.debug(f"    - {item_path}: korean_style 정의를 추가했습니다.")

                # 변경된 경우에만 수정된 내용 저장, 아니면 원본 저장
                if css_changed:
                    modified_css_content[item_path] = css_content.encode(detected_encoding)
                else:
                    modified_css_content[item_path] = item_bytes # 원본 바이트 그대로

            except Exception as css_proc_err:
                logging.error(f"CSS 파일 처리 중 오류 발생 ({item_path}): {css_proc_err}", exc_info=True)
                # 오류 발생 시 해당 파일은 원본 내용 유지
                if item_path not in modified_css_content:
                    modified_css_content[item_path] = original_content_map.get(item_path, b'')

    logging.info(f"CSS 파일 처리 완료. {len(modified_css_content)}개의 CSS 파일 처리 결과 저장됨.")
    return modified_css_content


def update_opf_metadata(opf_soup, updated_metadata):
    metadata = opf_soup.find('metadata')
    if not metadata:
        logging.error("<metadata> 태그를 OPF에서 찾을 수 없습니다. 메타데이터 업데이트 건너뜁니다.")
        return opf_soup

    try:
        language_tag = metadata.find('dc:language')
        if language_tag:
            if language_tag.string != 'ko':
                language_tag.string = 'ko'
                logging.debug("OPF 언어 태그 'ko' 업데이트.")
        else:
            new_lang_tag = opf_soup.new_tag("dc:language")
            new_lang_tag.string = "ko"
            date_tag = metadata.find('dc:date')
            if date_tag:
                date_tag.insert_before(new_lang_tag)
            else:
                metadata.append(new_lang_tag)
            logging.debug("OPF 언어 태그 'ko' 추가.")

        translator_name = "AI Translator (Gemini)"
        contributor_tag = metadata.find('dc:contributor', string=translator_name)
        if not contributor_tag:
            new_contributor = opf_soup.new_tag("dc:contributor")
            new_contributor.string = translator_name
            new_contributor['opf:role'] = "trl"
            metadata.append(new_contributor)
            logging.debug(f"OPF 번역가 정보 '{translator_name}' 추가.")

        title_info = updated_metadata.get('title')
        if title_info and 'value' in title_info:
            title_value = title_info['value']
            title_id = title_info.get('id')
            title_tag = metadata.find('dc:title', id=title_id) if title_id else metadata.find('dc:title')

            if title_tag:
                if title_tag.string != title_value:
                    title_tag.string = title_value
                    logging.debug(f"OPF dc:title 업데이트: '{title_value}'")

                if title_id:
                    file_as_meta_title = metadata.find('meta', attrs={'property': 'file-as', 'refines': f'#{title_id}'})
                    if file_as_meta_title:
                        if file_as_meta_title.string is None or file_as_meta_title.string.strip() != title_value:
                            file_as_meta_title.string = title_value
                            logging.debug(f"OPF title file-as meta *텍스트* 업데이트: '{title_value}'")
                    else:
                        new_file_as_meta = opf_soup.new_tag('meta', attrs={'property': 'file-as', 'refines': f'#{title_id}'})
                        new_file_as_meta.string = title_value
                        metadata.append(new_file_as_meta)
                        logging.debug(f"OPF title file-as meta 추가 (텍스트: '{title_value}')")
                else:
                    logging.warning("Title 태그에 ID가 없어 file-as meta 태그를 처리할 수 없습니다.")
            else:
                 logging.warning(f"업데이트할 dc:title 태그(ID: {title_id})를 찾을 수 없습니다.")


        creator_list = updated_metadata.get('creators', [])
        for creator_info in creator_list:
            if 'id' in creator_info and 'value' in creator_info:
                creator_id = creator_info['id']
                creator_value = creator_info['value']
                creator_tag = metadata.find('dc:creator', id=creator_id)

                if creator_tag:
                    if creator_tag.string != creator_value:
                        creator_tag.string = creator_value
                        logging.debug(f"OPF dc:creator (ID: {creator_id}) 업데이트: '{creator_value}'")

                    if creator_tag.has_attr('opf:file-as'):
                        if creator_tag['opf:file-as'] != creator_value:
                            creator_tag['opf:file-as'] = creator_value
                            logging.debug(f"OPF dc:creator (ID: {creator_id})의 opf:file-as 속성 업데이트: '{creator_value}'")

                    file_as_meta_creator = metadata.find('meta', attrs={'property': 'file-as', 'refines': f'#{creator_id}'})
                    if file_as_meta_creator:
                        if file_as_meta_creator.string is None or file_as_meta_creator.string.strip() != creator_value:
                            file_as_meta_creator.string = creator_value
                            logging.debug(f"OPF creator (ID: {creator_id}) file-as meta *텍스트* 업데이트: '{creator_value}'")
                    else:
                        new_file_as_meta = opf_soup.new_tag('meta', attrs={'property': 'file-as', 'refines': f'#{creator_id}'})
                        new_file_as_meta.string = creator_value
                        metadata.append(new_file_as_meta)
                        logging.debug(f"OPF creator (ID: {creator_id}) file-as meta 추가 (텍스트: '{creator_value}')")
                else:
                     logging.warning(f"업데이트할 dc:creator 태그(ID: {creator_id})를 찾을 수 없습니다.")

        publisher_info = updated_metadata.get('publisher')
        if publisher_info and 'value' in publisher_info:
            publisher_value = publisher_info['value']
            publisher_id = publisher_info.get('id')
            publisher_tag = metadata.find('dc:publisher', id=publisher_id) if publisher_id else metadata.find('dc:publisher')

            if publisher_tag:
                if publisher_tag.string != publisher_value:
                    publisher_tag.string = publisher_value
                    logging.debug(f"OPF dc:publisher 업데이트: '{publisher_value}'")

                if publisher_id:
                    file_as_meta_publisher = metadata.find('meta', attrs={'property': 'file-as', 'refines': f'#{publisher_id}'})
                    if file_as_meta_publisher:
                        if file_as_meta_publisher.string is None or file_as_meta_publisher.string.strip() != publisher_value:
                            file_as_meta_publisher.string = publisher_value
                            logging.debug(f"OPF publisher file-as meta *텍스트* 업데이트: '{publisher_value}'")
                    else:
                        new_file_as_meta = opf_soup.new_tag('meta', attrs={'property': 'file-as', 'refines': f'#{publisher_id}'})
                        new_file_as_meta.string = publisher_value
                        metadata.append(new_file_as_meta)
                        logging.debug(f"OPF publisher file-as meta 추가 (텍스트: '{publisher_value}')")
                else:
                    logging.warning("Publisher 태그에 ID가 없어 file-as meta 태그를 처리할 수 없습니다.")
            else:
                 logging.warning(f"업데이트할 dc:publisher 태그(ID: {publisher_id})를 찾을 수 없습니다.")


        title_sort_tag = metadata.find('meta', attrs={'name': 'calibre:title_sort'})
        if title_info and 'value' in title_info and title_info['value']:
            title_value_for_sort = title_info['value']
            if title_sort_tag:
                 if title_sort_tag.get('content') != title_value_for_sort:
                     title_sort_tag['content'] = title_value_for_sort
                     logging.debug(f"OPF calibre:title_sort 업데이트: '{title_value_for_sort}'")
            else:
                 new_sort_meta = opf_soup.new_tag('meta', attrs={'name': 'calibre:title_sort', 'content': title_value_for_sort})
                 metadata.append(new_sort_meta)
                 logging.debug(f"OPF calibre:title_sort 추가: '{title_value_for_sort}'")
        elif title_sort_tag is None:
            title_tag_existing = metadata.find('dc:title')
            if title_tag_existing and title_tag_existing.string:
                 existing_title_text = title_tag_existing.string.strip()
                 if existing_title_text:
                     new_sort_meta = opf_soup.new_tag('meta', attrs={'name': 'calibre:title_sort', 'content': existing_title_text})
                     metadata.append(new_sort_meta)
                     logging.debug(f"OPF calibre:title_sort 추가 (기존 제목 기반): '{existing_title_text}'")

    except Exception as e:
        logging.error(f"OPF 메타데이터 업데이트 중 오류 발생: {e}", exc_info=True)

    return opf_soup
    

def update_opf_manifest(opf_soup, font_filename, opf_dir):
    """
    OPF의 <manifest> 섹션에 필요한 항목(예: 사용자 정의 폰트)을 추가합니다.
    href 경로는 OPF 파일 기준의 상대 경로로 설정됩니다. ('Fonts/파일명' 형태)

    Args:
        opf_soup (BeautifulSoup object): 파싱된 OPF 내용.
        font_filename (str): 추가할 폰트 파일 이름.
        opf_dir (str): OPF 파일이 있는 EPUB 내부 디렉토리 경로 (루트 기준, 예: 'OEBPS' 또는 '').

    Returns:
        BeautifulSoup object: 수정된 opf_soup 객체.
    """
    manifest = opf_soup.find('manifest')
    if not manifest:
        logging.error("<manifest> 태그를 OPF에서 찾을 수 없습니다. 매니페스트 업데이트 건너뜁니다.")
        return opf_soup

    try:
        font_id = os.path.splitext(font_filename)[0]
        font_folder_name = "Fonts" # EPUB 내 폰트 폴더 이름

        # >>>>> 수정: OPF 기준 상대 경로 직접 생성 <<<<<
        # OPF 파일에서 Fonts 폴더 안의 폰트 파일을 가리키는 상대 경로
        # 예: OPF가 OEBPS/content.opf -> Fonts/RIDIBatang.otf
        # 예: OPF가 content.opf -> Fonts/RIDIBatang.otf
        font_href_relative = posixpath.join(font_folder_name, font_filename)
        # >>>>> 수정 끝 <<<<<

        # --- 디버깅 로그 추가 (값 확인용) ---
        logging.debug(f"update_opf_manifest DEBUG:")
        logging.debug(f"  - font_filename: {font_filename}")
        logging.debug(f"  - opf_dir: '{opf_dir}'")
        # logging.debug(f"  - font_abs_path_in_epub (참고): {posixpath.normpath(posixpath.join(opf_dir, font_folder_name, font_filename))}") # 참고용
        logging.debug(f"  - font_href_relative (used): {font_href_relative}")
        # --- 디버깅 로그 끝 ---

        # 파일 확장자에 따라 media-type 결정
        if font_filename.lower().endswith(".otf"):
            font_media_type = "application/vnd.ms-opentype"
        elif font_filename.lower().endswith(".ttf"):
            font_media_type = "application/font-sfnt"
        elif font_filename.lower().endswith(".woff"):
            font_media_type = "application/font-woff"
        elif font_filename.lower().endswith(".woff2"):
             font_media_type = "font/woff2"
        else:
            font_media_type = "application/octet-stream"
            logging.warning(f"알 수 없는 폰트 확장자: {font_filename}. media-type을 application/octet-stream으로 설정합니다.")

        # 매니페스트에 폰트 아이템이 이미 있는지 확인 (id 또는 계산된 상대 href 기준)
        existing_item_by_id = manifest.find('item', attrs={'id': font_id})
        existing_item_by_href = manifest.find('item', attrs={'href': font_href_relative})

        if not existing_item_by_id and not existing_item_by_href:
            # 새 폰트 아이템 추가 (계산된 상대 경로 사용)
            new_font_item = opf_soup.new_tag('item', id=font_id, href=font_href_relative, **{'media-type': font_media_type})
            manifest.append(new_font_item)
            logging.info(f"OPF 매니페스트에 폰트 항목(ID: {font_id}, Href: {font_href_relative})을 추가했습니다.")
        else:
             # 이미 존재하면 경고 또는 업데이트 로직 추가 가능
             existing_href = existing_item_by_id.get('href') if existing_item_by_id else existing_item_by_href.get('href')
             logging.debug(f"OPF 매니페스트에 폰트 항목(ID: {font_id} 또는 Href: {existing_href})이 이미 존재합니다.")
             # ID는 같은데 href가 다른 경우 업데이트
             if existing_item_by_id and existing_item_by_id.get('href') != font_href_relative:
                 logging.warning(f"Updating existing font item ID '{font_id}' href from '{existing_item_by_id.get('href')}' to '{font_href_relative}'")
                 existing_item_by_id['href'] = font_href_relative
             # Href는 같은데 ID가 다른 경우 (덜 일반적) - 로깅만 하거나 필요시 ID 업데이트
             elif existing_item_by_href and existing_item_by_href.get('id') != font_id:
                  logging.warning(f"Existing font item with href '{font_href_relative}' has different ID '{existing_item_by_href.get('id')}' (expected '{font_id}')")
                  # 필요시: existing_item_by_href['id'] = font_id


    except Exception as e:
        logging.error(f"OPF 매니페스트 업데이트 중 오류 발생: {e}", exc_info=True)

    return opf_soup

def finalize_opf(opf_soup):
    """
    수정된 BeautifulSoup OPF 객체를 최종 XML 문자열로 변환하고 UTF-8 바이트로 인코딩합니다.

    Args:
        opf_soup (BeautifulSoup object): 수정된 OPF 객체.

    Returns:
        bytes: 최종 OPF 파일 내용 (UTF-8 인코딩된 바이트). 오류 시 None 반환.
    """
    try:
        # BeautifulSoup 객체를 문자열로 변환 (prettify는 사용하지 않거나 선택적으로 사용)
        # prettify는 불필요한 공백을 추가할 수 있으므로 str() 사용 권장
        opf_str_final = str(opf_soup)

        # UTF-8 바이트로 인코딩
        opf_bytes = opf_str_final.encode('utf-8')
        logging.debug("OPF 객체를 UTF-8 바이트로 변환 완료.")
        return opf_bytes
    except Exception as e:
        logging.error(f"OPF 최종 변환 및 인코딩 중 오류 발생: {e}", exc_info=True)
        return None


def reconstruct_translated_xhtml(xhtml_base_name, json_data, output_dir, mode=None):
    reconstructed_xhtml = ""
    before_body_str = ""
    body_parts = []
    end_body_str = ""

    if xhtml_base_name not in json_data:
        logging.error(f"JSON 데이터에서 XHTML 파일 '{xhtml_base_name}' 정보를 찾을 수 없습니다.")
        return None

    try:
        for block in json_data[xhtml_base_name]:
            block_type = block.get("type")
            content_filepath = block.get("content")

            if not block_type or not content_filepath:
                logging.warning(f"XHTML '{xhtml_base_name}'의 블록 정보가 유효하지 않습니다: {block}")
                continue

            if not os.path.isabs(content_filepath):
                 content_filepath = os.path.join(output_dir, content_filepath)

            if not os.path.exists(content_filepath):
                logging.warning(f"XHTML '{xhtml_base_name}' 재구성 중 파일 누락: {content_filepath}. 건너뜁니다.")
                continue

            content_part = None
            try:
                with open(content_filepath, 'r', encoding='utf-8') as cf:
                    content_part = cf.read()
            except UnicodeDecodeError:
                try:
                    with open(content_filepath, 'r', encoding='cp949') as cf:
                        content_part = cf.read()
                        logging.info(f"파일 '{content_filepath}'을(를) cp949로 읽었습니다.")
                except Exception as read_err_inner:
                    logging.error(f"파일 읽기 실패 ({content_filepath}): {read_err_inner}")
                    content_part = f"<!-- Read Error: {os.path.basename(content_filepath)} -->"
            except Exception as read_err_outer:
                logging.error(f"파일 읽기 중 오류 ({content_filepath}): {read_err_outer}")
                content_part = f"<!-- Read Error: {os.path.basename(content_filepath)} -->"

            if content_part is None: continue

            # ### 이 부분이 올바른 if/elif 구조입니다. ###
            if block_type == "before_body":
                before_body_str = content_part
            
            # --- START: 여기가 수정된 핵심 로직입니다. ---
            elif block_type.startswith("text_block"):
                processed_lines = []
                for line in content_part.splitlines():
                    stripped_line = line.strip()
                    if stripped_line:
                        # 줄이 HTML 태그(예: <ruby>)로 시작하는지 확인
                        is_html_like = re.match(r'^\s*<.+>', stripped_line, re.DOTALL)
                        
                        if is_html_like:
                            # 이미 태그가 있으면 그대로 추가
                            processed_lines.append(stripped_line) 
                        else:
                            # 순수 텍스트 줄에만 <p> 태그 추가
                            processed_lines.append(f'<p class="korean_style">{stripped_line}</p>')
                
                body_parts.append("\n".join(processed_lines))
            # --- END: 수정된 핵심 로직 끝 ---
            
            elif block_type == "image":
                body_parts.append(content_part)
            elif block_type == "end_body":
                end_body_str = content_part
            else:
                logging.warning(f"알 수 없는 블록 타입: {block_type} in {xhtml_base_name}")
                body_parts.append(f"<!-- Unknown block type: {block_type} -->\n{content_part}")

        reconstructed_xhtml = before_body_str + "\n" + "\n".join(body_parts) + "\n" + end_body_str
        logging.debug(f"XHTML '{xhtml_base_name}' 내용 재구성 완료.")

        try:
            soup = BeautifulSoup(reconstructed_xhtml, 'html.parser')
            html_tag = soup.find('html')
            if html_tag:
                if html_tag.get('lang') != 'ko':
                    html_tag['lang'] = 'ko'
                    logging.debug(f"    - {xhtml_base_name}: html lang 속성을 'ko'로 설정.")
                if 'xmlns' in html_tag.attrs and html_tag['xmlns'] == "http://www.w3.org/1999/xhtml":
                    if html_tag.get('xml:lang') != 'ko':
                         html_tag['xml:lang'] = 'ko'
                         logging.debug(f"    - {xhtml_base_name}: html xml:lang 속성을 'ko'로 설정.")
                elif html_tag.get('xml:lang') and html_tag.get('xml:lang') != 'ko':
                     html_tag['xml:lang'] = 'ko'
                     logging.debug(f"    - {xhtml_base_name}: html xml:lang 속성을 'ko'로 설정 (네임스페이스 없음).")

            final_xhtml_str = str(soup)
            xhtml_bytes = final_xhtml_str.encode('utf-8')
            return xhtml_bytes

        except Exception as soup_proc_err:
            logging.error(f"XHTML 파싱 또는 최종 처리 중 오류 ({xhtml_base_name}): {soup_proc_err}", exc_info=True)
            return reconstructed_xhtml.encode('utf-8')

    except Exception as recon_err:
        logging.error(f"XHTML 재구성 중 예외 발생 ({xhtml_base_name}): {recon_err}", exc_info=True)
        return None


def process_ncx(ncx_bytes, updated_metadata, nav_translations, model, api_call_delay, last_api_call_time_ref):
    ncx_modified = False
    ncx_encoding = 'utf-8'

    try:
        NS = {'ncx': 'http://www.daisy.org/z3986/2005/ncx/'}
        ET.register_namespace('', NS['ncx'])

        ncx_bytes_no_bom = ncx_bytes[3:] if ncx_bytes.startswith(b'\xef\xbb\xbf') else ncx_bytes
        ncx_str = None
        try:
            ncx_str = ncx_bytes_no_bom.decode(ncx_encoding)
        except UnicodeDecodeError:
            try:
                ncx_encoding = 'cp949'
                ncx_str = ncx_bytes_no_bom.decode(ncx_encoding)
                logging.info(f"NCX 파일을 {ncx_encoding}으로 디코딩했습니다.")
            except Exception as decode_err_inner:
                logging.error(f"NCX 파일 디코딩 최종 실패: {decode_err_inner}. 원본 바이트 사용.")
                return ncx_bytes
        except Exception as decode_err_outer:
             logging.error(f"NCX 파일 디코딩 중 오류: {decode_err_outer}. 원본 바이트 사용.")
             return ncx_bytes

        if ncx_str is None: return ncx_bytes

        tree = ET.fromstring(ncx_str)
        ncx_tag = tree

        lang_attr = '{http://www.w3.org/XML/1998/namespace}lang'
        if ncx_tag.tag == '{' + NS['ncx'] + '}ncx':
            if ncx_tag.get(lang_attr) != 'ko':
                ncx_tag.set(lang_attr, 'ko')
                ncx_modified = True
                logging.debug("NCX 루트 태그의 xml:lang 속성을 'ko'로 업데이트했습니다.")
        else:
            logging.warning("NCX 파일의 루트 태그가 예상과 다릅니다.")

        title_info = updated_metadata.get('title')
        if title_info and 'value' in title_info:
            meta_title_value = title_info['value']
            for doctitle in tree.findall('.//ncx:docTitle', NS):
                text_tag = doctitle.find('ncx:text', NS)
                if text_tag is not None and meta_title_value and text_tag.text != meta_title_value:
                    text_tag.text = meta_title_value
                    ncx_modified = True
                    logging.debug(f"NCX docTitle을 '{meta_title_value}'(으)로 업데이트했습니다.")

        items_translated_count = 0
        logging.info("NCX 내비게이션 항목 번역 시작...")
        for navLabel in tree.findall('.//ncx:navMap//ncx:navLabel', NS):
            text_tag = navLabel.find('ncx:text', NS)
            if text_tag is not None and text_tag.text:
                original_nav_text = text_tag.text.strip()
                if not original_nav_text: continue

                translated_text_dict = nav_translations.get(original_nav_text)
                translated_text_final = None
                translation_source = None

                if translated_text_dict and translated_text_dict != original_nav_text:
                    translated_text_final = translated_text_dict
                    translation_source = 'Dictionary'
                elif not translated_text_dict:
                    current_time = time.time()
                    time_since_last_call = current_time - last_api_call_time_ref[0]
                    if time_since_last_call < api_call_delay:
                        sleep_time = api_call_delay - time_since_last_call
                        time.sleep(sleep_time)

                    try:
                        nav_translation_prompt = "Translate the following Japanese navigation menu item text to Korean. Output only the Korean translation, without any explanations or quotation marks: '{text}'"
                        prompt = nav_translation_prompt.format(text=original_nav_text)
                        response = model.generate_content(prompt)
                        last_api_call_time_ref[0] = time.time()
                        translated_text_api_raw = response.text.strip()
                        translated_text_api = translated_text_api_raw.strip('"\'')

                        if translated_text_api and translated_text_api != original_nav_text:
                            translated_text_final = translated_text_api
                            translation_source = 'API'
                        else:
                            logging.warning(f"API translation failed or no change (NCX Nav: '{original_nav_text}'). Raw result: '{translated_text_api_raw}'")

                    except InvalidArgument as iae:
                         last_api_call_time_ref[0] = time.time()
                         logging.error(f"NCX Nav 항목 '{original_nav_text}' API 호출 중 InvalidArgument 오류: {iae}")
                    except Exception as api_err:
                        last_api_call_time_ref[0] = time.time()
                        logging.error(f"NCX 내비게이션 항목 '{original_nav_text}' API 번역 중 오류: {api_err}")

                if translated_text_final:
                    try:
                        text_tag.text = translated_text_final
                        ncx_modified = True
                        items_translated_count += 1
                    except Exception as replace_err:
                        logging.error(f"NCX navLabel 텍스트 교체 중 오류 ('{original_nav_text}'): {replace_err}")

        if ncx_modified:
            logging.info(f"NCX 파일 수정 완료 ({items_translated_count}개 항목 번역됨). 직렬화 중...")
            raw_string = ET.tostring(tree, encoding=ncx_encoding, xml_declaration=True)
            parsed_xml = minidom.parseString(raw_string)
            pretty_xml_lines = parsed_xml.toprettyxml(indent="  ").splitlines()
            pretty_xml_no_blank_lines = [line for line in pretty_xml_lines if line.strip()]
            final_ncx_bytes = "\n".join(pretty_xml_no_blank_lines).encode(ncx_encoding)
            return final_ncx_bytes
        else:
            logging.info("NCX 파일에 변경 사항이 없어 원본 내용을 사용합니다.")
            return ncx_bytes

    except ET.ParseError as xml_err:
        logging.error(f"NCX 파일 XML 파싱 오류: {xml_err}. 원본 바이트 사용.")
        return ncx_bytes
    except Exception as ncx_proc_err:
        logging.error(f"NCX 처리 중 예외 발생: {ncx_proc_err}", exc_info=True)
        if 'ncx_bytes' in locals() and ncx_bytes is not None:
             return ncx_bytes
        else:
             return b''
             

def process_nav_doc(nav_doc_bytes, updated_metadata, model, api_call_delay, last_api_call_time_ref, mode=None):
    nav_modified = False
    nav_encoding = 'utf-8'

    try:
        nav_bytes_no_bom = nav_doc_bytes[3:] if nav_doc_bytes.startswith(b'\xef\xbb\xbf') else nav_doc_bytes
        nav_content = None
        try:
            nav_content = nav_bytes_no_bom.decode(nav_encoding)
        except UnicodeDecodeError:
            try:
                nav_encoding = 'cp949'
                nav_content = nav_bytes_no_bom.decode(nav_encoding)
                logging.info(f"Nav Doc 파일을 {nav_encoding}으로 디코딩했습니다.")
            except Exception as decode_err_inner:
                logging.error(f"Nav Doc 파일 디코딩 최종 실패: {decode_err_inner}. 원본 바이트 사용.")
                return nav_doc_bytes
        except Exception as decode_err_outer:
             logging.error(f"Nav Doc 파일 디코딩 중 오류: {decode_err_outer}. 원본 바이트 사용.")
             return nav_doc_bytes

        if nav_content is None: return nav_doc_bytes

        soup_nav = BeautifulSoup(nav_content, 'html.parser')
        content_modified_by_soup = False

        html_tag_nav = soup_nav.find('html')
        if html_tag_nav:
            lang_changed = False
            if html_tag_nav.get('lang') != 'ko':
                html_tag_nav['lang'] = 'ko'
                lang_changed = True
            if 'xmlns' in html_tag_nav.attrs and html_tag_nav['xmlns'] == "http://www.w3.org/1999/xhtml":
                 if html_tag_nav.get('xml:lang') != 'ko':
                     html_tag_nav['xml:lang'] = 'ko'
                     lang_changed = True
            elif html_tag_nav.get('xml:lang') and html_tag_nav.get('xml:lang') != 'ko':
                 html_tag_nav['xml:lang'] = 'ko'
                 lang_changed = True

            if lang_changed:
                content_modified_by_soup = True
                logging.debug("Nav Doc 언어 속성 업데이트됨.")


        title_tag_nav = soup_nav.find('title')
        title_info = updated_metadata.get('title')
        if title_info and 'value' in title_info:
            meta_title_value = title_info['value']
            if title_tag_nav and meta_title_value and title_tag_nav.string != meta_title_value:
                title_tag_nav.string = meta_title_value
                content_modified_by_soup = True
                logging.debug(f"Nav Doc title 업데이트됨: {meta_title_value}")
        elif title_tag_nav and updated_metadata.get('title') is None:
             if title_tag_nav.string:
                  title_tag_nav.string = ""
                  content_modified_by_soup = True
                  logging.debug("Nav Doc title이 빈 값으로 업데이트됨.")


        items_translated_count = 0
        nav_items = soup_nav.select('nav[epub\\:type~="toc"] li a, nav[epub\\:type~="landmarks"] li a, nav[role="doc-toc"] li a')
        if not nav_items:
             nav_items = soup_nav.select('nav li a')
        if not nav_items:
             nav_items = soup_nav.select('body li a')


        if nav_items:
             logging.info("Nav Doc 내비게이션 항목 번역 시작...")
             global NAV_TRANSLATIONS
             nav_translation_prompt = "Translate the following Japanese navigation menu item text to Korean. Output only the Korean translation, without any explanations or quotation marks: '{text}'"

             for item_tag in nav_items:
                 target_node = None
                 potential_texts = [s for s in item_tag.strings if str(s).strip()]
                 if potential_texts:
                     target_node_content = potential_texts[0].strip()
                     for node in item_tag.find_all(string=True, recursive=True):
                          if isinstance(node, NavigableString) and node.strip() == target_node_content:
                              target_node = node
                              break

                 if not target_node: continue

                 original_text = str(target_node).strip()
                 if not original_text: continue

                 translated_text_final = None
                 translation_source = None

                 translated_text_dict = NAV_TRANSLATIONS.get(original_text)
                 if translated_text_dict is not None and translated_text_dict != original_text:
                     translated_text_final = translated_text_dict
                     translation_source = 'Dictionary'

                 is_api_condition_met = translated_text_dict is None

                 if is_api_condition_met:
                     current_time = time.time()
                     time_since_last_call = current_time - last_api_call_time_ref[0]
                     if time_since_last_call < api_call_delay:
                         sleep_time = api_call_delay - time_since_last_call
                         time.sleep(sleep_time)

                     try:
                         prompt = nav_translation_prompt.format(text=original_text)
                         response = model.generate_content(prompt)
                         last_api_call_time_ref[0] = time.time()
                         translated_text_api_raw = response.text
                         translated_text_api = translated_text_api_raw.strip().strip('"\'')

                         if translated_text_api and translated_text_api != original_text:
                             translated_text_final = translated_text_api
                             translation_source = 'API'
                         else:
                             logging.warning(f"API translation failed or no change (Nav Doc: '{original_text}'). Raw result: '{translated_text_api_raw}'")

                     except InvalidArgument as iae:
                          last_api_call_time_ref[0] = time.time()
                          logging.error(f"Nav Doc 항목 '{original_text}' API 호출 중 InvalidArgument 오류: {iae}")
                     except Exception as api_err:
                         last_api_call_time_ref[0] = time.time()
                         logging.error(f"Nav Doc 항목 '{original_text}' API 번역 중 오류: {api_err}")

                 if translated_text_final:
                     try:
                         target_node.replace_with(NavigableString(translated_text_final))
                         content_modified_by_soup = True
                         items_translated_count += 1
                     except Exception as replace_err:
                         logging.error(f"Nav Doc 텍스트 교체 중 오류 ('{original_text}' -> '{translated_text_final}'): {replace_err}")
        else:
            logging.info("Nav Doc에서 번역 대상 내비게이션 항목을 찾지 못했습니다.")

        modified_content_str = str(soup_nav)
        nav_modified = content_modified_by_soup

        if nav_modified:
            log_message = "Nav Doc 파일 수정 완료"
            if items_translated_count > 0: log_message += f" ({items_translated_count}개 항목 번역됨)"
            if not items_translated_count and content_modified_by_soup: log_message += " (내용 변경)"
            log_message += ". 최종 인코딩 중..."
            logging.info(log_message)
            try:
                final_nav_bytes = modified_content_str.encode('utf-8')
                return final_nav_bytes
            except Exception as enc_err:
                 logging.error(f"Nav Doc 최종 인코딩 중 오류: {enc_err}. 원본 바이트 반환.")
                 return nav_doc_bytes
        else:
            logging.info("Nav Doc 파일에 변경 사항이 없어 원본 내용을 사용합니다.")
            return nav_doc_bytes

    except Exception as nav_proc_err:
        logging.error(f"Nav Doc 처리 중 예외 발생: {nav_proc_err}", exc_info=True)
        if 'nav_doc_bytes' in locals() and nav_doc_bytes is not None:
            return nav_doc_bytes
        else:
            return b''
            

def process_cover_image(cover_image_bytes, cover_image_modify, cover_text_position, cover_text, font_path, font_size, font_color, background_color, script_dir):
    """
    설정에 따라 커버 이미지(바이트)에 텍스트를 오버레이합니다.

    Args:
        cover_image_bytes (bytes): 원본 커버 이미지 내용.
        cover_image_modify (int): 커버 수정 여부 (1: 수정, 2: 미수정).
        cover_text_position (int): 텍스트 위치 (1:TL, 2:TR, 3:BL, 4:BR).
        cover_text (str): 오버레이할 텍스트.
        font_path (str): 폰트 파일 경로 (절대 또는 상대).
        font_size (int): 폰트 크기.
        font_color (str): 폰트 색상 (16진수 RGB, 예: "FFFFFF").
        background_color (str): 배경 색상 (16진수 RGB, 예: "FF0000").
        script_dir (str): 스크립트 실행 디렉토리 경로 (상대 폰트 경로 해석용).

    Returns:
        bytes: 수정된 커버 이미지 바이트 (또는 수정 안 됐으면 원본 바이트).
    """
    if cover_image_modify != 1:
        logging.debug("커버 이미지 수정 설정이 비활성화되어 원본 이미지를 사용합니다.")
        return cover_image_bytes

    try:
        logging.info("커버 이미지 수정 시작...")
        image = Image.open(io.BytesIO(cover_image_bytes))
        # RGBA 이미지를 RGB로 변환 (JPEG 저장 호환성 및 투명도 제거)
        if image.mode == 'RGBA':
             image = image.convert('RGB')
        draw = ImageDraw.Draw(image)

        # 폰트 로드
        try:
            # 폰트 경로 처리 (절대/상대)
            effective_font_path = font_path if os.path.isabs(font_path) else os.path.join(script_dir, font_path)
            if not os.path.exists(effective_font_path):
                 raise FileNotFoundError(f"폰트 파일을 찾을 수 없습니다: {effective_font_path}")
            font = ImageFont.truetype(effective_font_path, int(font_size))
            logging.debug(f"폰트 로드 성공: {effective_font_path}")
        except Exception as font_err:
            font = ImageFont.load_default() # 기본 폰트 사용 (Fallback)
            logging.warning(f"폰트 로드 실패 ('{font_path}'): {font_err}. 기본 폰트를 사용합니다.")

        # 텍스트 및 박스 크기 계산
        text_to_draw = cover_text if cover_text else "AI 번역본"
        # textbbox 사용 (Pillow 최신 버전 권장)
        try:
             bbox = draw.textbbox((0, 0), text_to_draw, font=font)
             # bbox = (left, top, right, bottom)
             text_width = bbox[2] - bbox[0]
             text_height = bbox[3] - bbox[1]
             # top은 베이스라인 위쪽 여백 등을 포함할 수 있으므로, 실제 높이는 bottom - top
        except AttributeError: # 구 버전 호환성 (textsize)
             logging.warning("textbbox 사용 불가. textsize 사용 (정확도 낮을 수 있음). Pillow 업데이트 권장.")
             text_width, text_height = draw.textsize(text_to_draw, font=font)
             bbox = (0, 0, text_width, text_height) # bbox 임의 생성 (top 값 부정확)


        padding = max(int(font_size * 0.15), 5) # 폰트 크기의 15% 또는 최소 5px 패딩
        box_width = text_width + 2 * padding
        box_height = text_height + 2 * padding
        margin = padding # 이미지 가장자리에서의 간격

        # 텍스트 박스 위치 결정
        if cover_text_position == 1: # Top Left
            box_x, box_y = margin, margin
        elif cover_text_position == 2: # Top Right
            box_x, box_y = image.width - box_width - margin, margin
        elif cover_text_position == 3: # Bottom Left
            box_x, box_y = margin, image.height - box_height - margin
        else: # Bottom Right (기본값)
            box_x, box_y = image.width - box_width - margin, image.height - box_height - margin

        # 실제 텍스트 그리기 위치 계산 (textbbox[1]은 top 값)
        # textbbox는 (0,0) 기준 좌표이므로, 실제 그릴 때는 box_x/y를 더하고 top 오프셋(bbox[1])을 빼줌
        text_x = box_x + padding
        text_y = box_y + padding - bbox[1]

        # 색상 파싱 함수
        def parse_color(color_str, default_color):
            color_str = color_str.lstrip('#')
            if len(color_str) == 6:
                try:
                    return tuple(int(color_str[i:i+2], 16) for i in (0, 2, 4))
                except ValueError:
                    logging.warning(f"잘못된 색상 코드: '{color_str}'. 기본값 사용.")
                    return default_color
            else:
                logging.warning(f"잘못된 색상 코드 길이: '{color_str}'. 기본값 사용.")
                return default_color

        # 색상 적용 (RGB)
        back_rgb = parse_color(background_color, (255, 0, 0)) # 기본 빨강
        font_rgb = parse_color(font_color, (255, 255, 255)) # 기본 흰색

        # 사각형 배경 그리기
        draw.rectangle([box_x, box_y, box_x + box_width, box_y + box_height], fill=back_rgb)
        # 텍스트 그리기
        draw.text((text_x, text_y), text_to_draw, font=font, fill=font_rgb)

        # 수정된 이미지를 바이트로 저장
        img_byte_arr = io.BytesIO()
        save_format = image.format or 'PNG' # 원본 포맷 유지 시도, 없으면 PNG
        if save_format.upper() == 'JPEG':
            image.save(img_byte_arr, format='JPEG', quality=95) # JPEG 품질 설정
        elif save_format.upper() == 'WEBP':
             # WebP 저장 옵션 (손실/무손실 등) - 여기서는 무손실 시도
             try:
                  image.save(img_byte_arr, format='WEBP', lossless=True)
             except Exception as webp_err:
                  logging.warning(f"WebP 무손실 저장 실패 ({webp_err}). 손실 압축 시도.")
                  image.save(img_byte_arr, format='WEBP', quality=90) # 손실 압축으로 재시도
        else:
            # PNG 또는 기타 형식 저장
             if image.mode == 'P': # 팔레트 모드일 경우 PNG 저장 시 오류 발생 가능성 체크 (선택적)
                  logging.debug("이미지 모드가 'P'(팔레트)입니다. PNG 저장 시 색상 문제 가능성 있음.")
             try:
                  image.save(img_byte_arr, format=save_format)
             except KeyError: # 지원하지 않는 포맷일 경우 PNG로 저장
                  logging.warning(f"지원하지 않는 이미지 포맷: {save_format}. PNG로 저장합니다.")
                  image.save(img_byte_arr, format='PNG')


        modified_image_bytes = img_byte_arr.getvalue()
        logging.info("커버 이미지 수정 완료.")
        return modified_image_bytes

    except FileNotFoundError as e: # 폰트 파일 못찾는 경우
         logging.error(f"커버 이미지 처리 오류: {e}")
         return cover_image_bytes # 원본 반환
    except Exception as img_err:
        logging.error(f"커버 이미지 처리 중 예외 발생: {img_err}", exc_info=True)
        return cover_image_bytes # 오류 시 원본 반환


def add_font_file(zout, font_path, script_dir, opf_dir):
    """
    지정된 폰트 파일을 찾아서 EPUB 아카이브(zout)의 Fonts/ 디렉토리에 추가합니다.

    Args:
        zout (zipfile.ZipFile): 쓰기용으로 열린 ZipFile 객체.
        font_path (str): 추가할 폰트 파일 경로 (설정 파일 기준).
        script_dir (str): 스크립트 실행 디렉토리.
        opf_dir (str): OPF 파일이 있는 EPUB 내부 디렉토리 경로.

    Returns:
        str or None: 추가된 폰트의 EPUB 내부 절대 경로, 실패 시 None.
    """
    font_filename = os.path.basename(font_path)
    # 스크립트 디렉토리 기준 폰트 소스 경로
    font_source_path = font_path if os.path.isabs(font_path) else os.path.join(script_dir, font_path)

    if os.path.exists(font_source_path):
        # EPUB 내부 대상 경로 (OPF 기준 상대 경로 사용)
        font_dest_rel_path = posixpath.join("Fonts", font_filename)
        # EPUB 루트 기준 절대 경로
        font_dest_abs_path = posixpath.normpath(posixpath.join(opf_dir, font_dest_rel_path))

        try:
            with open(font_source_path, 'rb') as font_file:
                font_data = font_file.read()
            # EPUB 아카이브에 추가
            zout.writestr(font_dest_abs_path, font_data)
            logging.info(f"EPUB에 폰트 파일 추가 완료: {font_dest_abs_path}")
            return font_dest_abs_path
        except Exception as font_copy_e:
            print_colored(f"오류: 폰트 파일 '{font_source_path}' 추가 실패: {font_copy_e}", colorama.Fore.RED)
            logging.error(f"폰트 파일 '{font_source_path}' 추가 실패: {font_copy_e}", exc_info=True)
            return None
    else:
        print_colored(f"오류: 폰트 파일 '{font_source_path}'을(를) 찾을 수 없습니다.", colorama.Fore.RED)
        logging.error(f"폰트 파일 '{font_source_path}'을(를) 찾을 수 없습니다.")
        return None

def write_epub_archive(output_path, original_content_map, original_zipinfo_map, processed_content_map, font_path, script_dir, opf_dir, mode):
    processed_paths_in_zip = set()

    try:
        logging.info(f"최종 EPUB 파일 생성 시작: {output_path} (Mode: {mode})")
        with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zout:
            mimetype_path = 'mimetype'
            # ★★★ mimetype 처리 수정: processed_content_map 에 있는지 확인 ★★★
            if mimetype_path in processed_content_map:
                mimetype_content = processed_content_map[mimetype_path]
                original_mimetype_info = original_zipinfo_map.get(mimetype_path)

                zi_mimetype = zipfile.ZipInfo(mimetype_path)
                if original_mimetype_info:
                    zi_mimetype.date_time = original_mimetype_info.date_time
                    zi_mimetype.external_attr = original_mimetype_info.external_attr if original_mimetype_info.external_attr else (0o600 << 16)
                zi_mimetype.compress_type = zipfile.ZIP_STORED

                zout.writestr(zi_mimetype, mimetype_content)
                processed_paths_in_zip.add(mimetype_path)
                logging.debug("mimetype 파일 기록 완료.")
            else:
                # 오류 메시지는 rebuild_epub_orchestrator 에서 먼저 출력될 것임
                logging.error("write_epub_archive: 처리된 콘텐츠 맵에서 mimetype을 찾을 수 없습니다!")
                # 유효성 위해 여기서 중단할 수도 있음
                # return None

            logging.info("EPUB 파일 내용 기록 중...")
            # ★★★ processed_content_map 기준 반복 ★★★
            for item_path, content_bytes in tqdm(processed_content_map.items(), desc="EPUB 파일 쓰는 중", unit="개", leave=False):
                if item_path == mimetype_path: continue

                original_info = original_zipinfo_map.get(item_path)
                zi = zipfile.ZipInfo(item_path) # ★★★ 키(item_path)가 최종 경로가 됨 ★★★

                if original_info:
                    zi.date_time = original_info.date_time
                    if not original_info.is_dir():
                       zi.external_attr = original_info.external_attr if original_info.external_attr else (0o600 << 16)
                    else:
                       zi.external_attr = (0o600 << 16)
                else:
                    # toc.ncx 등 새로 추가된 파일의 기본 속성
                    zi.external_attr = (0o600 << 16)

                zi.compress_type = zipfile.ZIP_DEFLATED

                try:
                    zout.writestr(zi, content_bytes)
                    processed_paths_in_zip.add(item_path)
                except Exception as write_err:
                    logging.error(f"파일 쓰기 오류 ({item_path}): {write_err}", exc_info=True)

            added_font_path = add_font_file(zout, font_path, script_dir, opf_dir)
            if added_font_path:
                 processed_paths_in_zip.add(added_font_path)

        logging.info(f"EPUB 파일 생성 완료: {output_path}")
        return output_path

    except Exception as e:
        print_colored(f"오류: EPUB 아카이브 생성 중 예외 발생: {e}", colorama.Fore.RED)
        logging.error(f"EPUB 아카이브 생성 중 예외 발생: {e}", exc_info=True)
        if os.path.exists(output_path):
            try:
                os.remove(output_path)
                logging.info(f"실패로 인해 부분적으로 생성된 EPUB 파일 삭제: {output_path}")
            except OSError as del_err:
                logging.warning(f"실패한 EPUB 파일 삭제 중 오류: {del_err}")
        return None


def convert_opf_to_epub2_standard(opf_soup_original, updated_metadata):
    """
    주어진 OPF BeautifulSoup 객체를 EPUB 2 표준에 가깝게 변환합니다.

    Args:
        opf_soup_original (BeautifulSoup object): 원본 (수정될 수 있는) OPF soup 객체.
        updated_metadata (dict): 업데이트된 메타데이터 ('title', 'identifier_uuid', cover_item_id 등 포함 가능).

    Returns:
        bytes: EPUB 2 형식으로 변환된 OPF 내용 (UTF-8 bytes), 실패 시 None.
    """
    try:
        logging.info("OPF를 EPUB 2 표준으로 변환 시도 중...")
        # 원본 수정을 방지하기 위해 깊은 복사 사용
        opf_soup = copy.deepcopy(opf_soup_original)

        # --- Package Tag ---
        package_tag = opf_soup.find('package')
        if not package_tag:
            logging.error("EPUB 2 변환 실패: <package> 태그를 찾을 수 없습니다.")
            return None

        package_tag['version'] = '2.0'
        # unique-identifier 설정 (기본값 또는 메타데이터에서 가져오기)
        unique_id_name = "BookId" # OPF 표준에서 권장하는 식별자 ID
        package_tag['unique-identifier'] = unique_id_name
        logging.debug(f"  - package version='2.0', unique-identifier='{unique_id_name}' 설정됨.")

        # --- Metadata Tag ---
        metadata_tag = opf_soup.find('metadata')
        if not metadata_tag:
            logging.error("EPUB 2 변환 실패: <metadata> 태그를 찾을 수 없습니다.")
            return None

        # EPUB 2 네임스페이스 추가/확인
        metadata_tag['xmlns:dc'] = "http://purl.org/dc/elements/1.1/"
        metadata_tag['xmlns:opf'] = "http://www.idpf.org/2007/opf"
        logging.debug("  - metadata 네임스페이스 (xmlns:dc, xmlns:opf) 확인/추가됨.")

        # dc:identifier 처리 (unique-identifier와 매칭되는 id 부여)
        identifier_tag = metadata_tag.find('dc:identifier')
        if identifier_tag:
            identifier_tag['id'] = unique_id_name
            # 만약 opf:scheme 속성이 있다면 유지하거나 필요에 따라 수정
            logging.debug(f"  - dc:identifier에 id='{unique_id_name}' 설정됨.")
        else:
            # 식별자가 없으면 새로 생성 (updated_metadata 활용)
            new_id_tag = opf_soup.new_tag('dc:identifier', id=unique_id_name)
            # 실제 UUID 값 등을 updated_metadata에서 가져와 설정하는 것이 좋음
            id_value = updated_metadata.get('identifier_uuid', f"urn:uuid:{uuid.uuid4()}")
            new_id_tag.string = id_value
            metadata_tag.append(new_id_tag)
            logging.debug(f"  - dc:identifier 생성됨 (id='{unique_id_name}', value='{id_value}').")

        # EPUB 3 전용 메타 태그 제거 (property 속성 사용 태그 등)
        epub3_meta_tags = metadata_tag.find_all('meta', property=True)
        for tag in epub3_meta_tags:
            logging.debug(f"  - EPUB 3 메타 태그 제거: {tag}")
            tag.decompose()

        # EPUB 3 link 태그 제거 (record 등)
        epub3_link_tags = metadata_tag.find_all('link', rel=True)
        for tag in epub3_link_tags:
             if 'record' in tag.get('rel', []): # refine 역할 등
                  logging.debug(f"  - EPUB 3 link 태그 제거: {tag}")
                  tag.decompose()


        # 커버 이미지 메타 태그 추가 (manifest에서 cover item ID 찾아야 함)
        manifest_tag = opf_soup.find('manifest')
        cover_item_id = None
        if manifest_tag:
             # Calibre 스타일 메타 태그 찾기
             cover_meta_calibre = metadata_tag.find('meta', attrs={'name': 'cover'})
             if cover_meta_calibre and cover_meta_calibre.get('content'):
                  cover_item_id = cover_meta_calibre['content']
             else:
             # 표준 EPUB 3 properties 속성 찾기 (이미 제거되었을 수 있음)
                 cover_item_original = opf_soup_original.find('manifest').find('item', attrs={'properties': re.compile(r'\bcover-image\b')})
                 if cover_item_original and cover_item_original.get('id'):
                      cover_item_id = cover_item_original['id']

        if cover_item_id:
            existing_cover_meta = metadata_tag.find('meta', attrs={'name': 'cover'})
            if existing_cover_meta:
                 if existing_cover_meta.get('content') != cover_item_id:
                     existing_cover_meta['content'] = cover_item_id
                     logging.debug(f"  - 기존 cover 메타 태그 업데이트: content='{cover_item_id}'")
            else:
                 new_cover_meta = opf_soup.new_tag('meta', attrs={'name': 'cover', 'content': cover_item_id})
                 metadata_tag.append(new_cover_meta)
                 logging.debug(f"  - cover 메타 태그 추가: content='{cover_item_id}'")
        else:
            logging.warning("  - 커버 이미지 ID를 찾지 못해 cover 메타 태그를 추가하지 못했습니다.")


        # --- Manifest Tag ---
        if manifest_tag:
            items_to_check = manifest_tag.find_all('item')
            for item in items_to_check:
                # EPUB 3 properties 속성 제거
                if item.has_attr('properties'):
                    logging.debug(f"  - manifest item에서 properties 속성 제거: ID={item.get('id')}")
                    del item['properties']
                # Fallback 속성 제거 (EPUB 3)
                if item.has_attr('fallback'):
                     logging.debug(f"  - manifest item에서 fallback 속성 제거: ID={item.get('id')}")
                     del item['fallback']
        else:
            logging.error("EPUB 2 변환 실패: <manifest> 태그를 찾을 수 없습니다.")
            return None

        # --- Spine Tag ---
        spine_tag = opf_soup.find('spine')
        if spine_tag:
            # toc="ncx" 는 RIDI 모드에서 이미 설정됨 (여기서는 확인만)
            if spine_tag.get('toc') != 'ncx':
                logging.warning("  - <spine> 태그에 toc='ncx' 속성이 없습니다 (RIDI 모드 로직 확인 필요).")
            # page-progression-direction 제거 (EPUB 3)
            if spine_tag.has_attr('page-progression-direction'):
                logging.debug("  - spine에서 page-progression-direction 속성 제거됨.")
                del spine_tag['page-progression-direction']
        else:
            logging.error("EPUB 2 변환 실패: <spine> 태그를 찾을 수 없습니다.")
            return None

        # --- Guide Tag (생성) ---
        guide_tag = opf_soup.find('guide')
        if not guide_tag:
            guide_tag = opf_soup.new_tag('guide')
            # spine 뒤 또는 다른 적절한 위치에 추가
            package_tag.append(guide_tag) # 간단히 package 끝에 추가
            logging.debug("  - <guide> 섹션 추가됨.")

            # 주요 랜드마크 추가 시도
            guide_items = {} # type -> href 매핑
            if cover_item_id: # 커버
                cover_item = manifest_tag.find('item', id=cover_item_id)
                if cover_item and cover_item.get('href'):
                     guide_items['cover'] = cover_item['href']

            # 목차 (NCX href 사용)
            ncx_item = manifest_tag.find('item', id='ncx') # toc="ncx"와 연결된 ID
            if ncx_item and ncx_item.get('href'):
                 guide_items['toc'] = ncx_item['href']

            # 제목 페이지 등 다른 랜드마크 추가 가능 (예: 첫번째 spine item)
            first_spine_itemref = spine_tag.find('itemref')
            if first_spine_itemref and first_spine_itemref.get('idref'):
                 first_item_id = first_spine_itemref['idref']
                 first_manifest_item = manifest_tag.find('item', id=first_item_id)
                 if first_manifest_item and first_manifest_item.get('href'):
                     guide_items.setdefault('title-page', first_manifest_item['href']) # 없으면 추가
                     guide_items.setdefault('text', first_manifest_item['href']) # 본문 시작점

            # guide 태그에 reference 추가
            for ref_type, ref_href in guide_items.items():
                 new_ref = opf_soup.new_tag('reference', type=ref_type, href=ref_href)
                 # title 속성은 필수는 아님
                 # new_ref['title'] = f"{ref_type.capitalize()}"
                 guide_tag.append(new_ref)
                 logging.debug(f"    - guide reference 추가: type='{ref_type}', href='{ref_href}'")

        # --- 최종 XML 생성 ---
        # soup 객체를 문자열로 변환 시 xml 선언 포함 및 UTF-8 인코딩
        # BeautifulSoup은 기본적으로 유효한 XML/HTML 구조를 유지하므로 추가적인 prettify는 선택사항
        final_opf_str = opf_soup.prettify(formatter="minimal") # 또는 str(opf_soup)

        # UTF-8 bytes 로 인코딩
        final_opf_bytes = final_opf_str.encode('utf-8')
        logging.info("OPF를 EPUB 2 표준 형식으로 변환 완료.")
        return final_opf_bytes

    except Exception as e:
        logging.error(f"OPF를 EPUB 2로 변환 중 오류 발생: {e}", exc_info=True)
        return None
        
        
def rebuild_epub_orchestrator(epub_path, json_data, updated_opf_soup, updated_metadata, model,
                             cover_image_modify, cover_text_position, cover_text, font_path, font_size,
                             font_color, background_color, translated_toc_map,
                             output_epub_path=None, mode='standard'):
    try:
        # --- (기존 output_dir 추정 로직 유지) ---
        script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
        epub_dir = os.path.dirname(epub_path)
        output_dir = None
        try:
             first_xhtml_key = next((k for k in json_data if k != "epub_filename"), None)
             if first_xhtml_key and isinstance(json_data[first_xhtml_key], list) and json_data[first_xhtml_key]:
                  first_block = json_data[first_xhtml_key][0]
                  if isinstance(first_block, dict) and 'content' in first_block and isinstance(first_block['content'], str):
                       content_path_check = first_block['content']
                       if os.path.isabs(content_path_check):
                            output_dir = os.path.dirname(content_path_check)
                       else:
                            # 상대 경로 처리 (스크립트 또는 EPUB 기준)
                            potential_path_script = os.path.join(script_dir, content_path_check)
                            if os.path.exists(os.path.dirname(potential_path_script)):
                                output_dir = os.path.dirname(potential_path_script)
                            else:
                                potential_path_epub = os.path.join(epub_dir, content_path_check)
                                if os.path.exists(os.path.dirname(potential_path_epub)):
                                     output_dir = os.path.dirname(potential_path_epub)
                                else:
                                      # Fallback to temp_translations directory structure
                                      epub_base_fallback = json_data.get("epub_filename", os.path.splitext(os.path.basename(epub_path))[0])
                                      temp_dir_fallback = os.path.join(script_dir, "temp_translations", epub_base_fallback)
                                      if os.path.isdir(temp_dir_fallback):
                                           output_dir = temp_dir_fallback


             if not output_dir or not os.path.isdir(output_dir):
                  # output_dir을 찾지 못했거나 유효하지 않은 경우 재시도/대체 경로 설정
                  epub_base = json_data.get("epub_filename", os.path.splitext(os.path.basename(epub_path))[0])
                  temp_translations_base = os.path.join(script_dir, "temp_translations")
                  output_dir_base = os.path.join(temp_translations_base, epub_base)
                  counter = 1
                  output_dir_final = output_dir_base
                  # 디렉토리를 찾거나 생성될 때까지 시도
                  while os.path.exists(output_dir_final):
                       if os.path.isdir(output_dir_final): # 찾았으면 사용
                            output_dir = output_dir_final
                            break
                       # 존재하지만 디렉토리가 아니면 다음 번호 시도
                       output_dir_final = f"{output_dir_base} ({counter})"
                       counter += 1
                  else: # 존재하지 않으면 이 경로 사용
                       output_dir = output_dir_final
                  logging.warning(f"output_dir 추정 실패 또는 경로 유효하지 않음. 임시 경로 사용 시도: {output_dir}")

        except Exception as e:
             # 예외 발생 시 기본 경로 사용
             epub_base = json_data.get("epub_filename", os.path.splitext(os.path.basename(epub_path))[0])
             temp_translations_base = os.path.join(script_dir, "temp_translations")
             output_dir_base = os.path.join(temp_translations_base, epub_base)
             # 폴더 존재 여부와 관계없이 일단 경로 지정
             output_dir = output_dir_base
             logging.error(f"output_dir 추정 중 오류 발생 ({e}). 기본 경로 사용 시도: {output_dir}")


        if not output_dir:
             print_colored("오류: 임시 작업 디렉토리를 결정할 수 없습니다.", colorama.Fore.RED)
             return None
        os.makedirs(output_dir, exist_ok=True)

        # 원본 EPUB 읽기 -> opf_dir 반환값 사용
        original_content_map, original_zipinfo_map, opf_path, opf_dir = read_original_epub(epub_path)
        if not original_content_map or not opf_path:
            return None

        opf_soup_copy = copy.deepcopy(updated_opf_soup)

        opf_soup_copy = update_opf_metadata(opf_soup_copy, updated_metadata)
        # >>>>> 수정: update_opf_manifest 호출 시 opf_dir 전달 <<<<<
        opf_soup_copy = update_opf_manifest(opf_soup_copy, os.path.basename(font_path), opf_dir)
        # >>>>> 수정 끝 <<<<<

        # --- (이후 로직은 이전 답변의 폴더 생성 문제 해결 코드와 동일하게 유지) ---
        final_opf_bytes = None
        ncx_target_path_in_epub = None

        if mode == 'ridi':
            # ... (RIDI 모드 OPF 처리 로직) ...
            logging.info("RIDI 모드: OPF 강제 수정 및 EPUB 2 변환 시도.")
            manifest = opf_soup_copy.find('manifest')
            spine = opf_soup_copy.find('spine')
            ncx_id = "ncx"
            ncx_href = "toc.ncx" # RIDI 표준 NCX 파일명
            ncx_target_path_in_epub = posixpath.normpath(posixpath.join(opf_dir, ncx_href)) # EPUB 루트 기준 경로

            if manifest and spine:
                nav_doc_item = manifest.find('item', attrs={'properties': re.compile(r'\bnav\b')})
                if nav_doc_item:
                    logging.debug(f"RIDI 모드: OPF manifest에서 Nav Doc 항목(ID: {nav_doc_item.get('id')}) 제거.")
                    nav_doc_item.decompose()

                ncx_item = manifest.find('item', attrs={'id': ncx_id})
                ncx_media_type = "application/x-dtbncx+xml"
                if ncx_item:
                    ncx_item['href'] = ncx_href
                    ncx_item['media-type'] = ncx_media_type
                    logging.debug(f"RIDI 모드: 기존 NCX 항목(ID: {ncx_id}) 정보 업데이트.")
                else:
                    new_ncx_item = opf_soup_copy.new_tag('item', id=ncx_id, href=ncx_href, **{'media-type': ncx_media_type})
                    manifest.append(new_ncx_item)
                    logging.debug(f"RIDI 모드: 새 NCX 항목(ID: {ncx_id}) 추가.")

                spine['toc'] = ncx_id
                logging.debug(f"RIDI 모드: Spine toc 속성을 '{ncx_id}'로 설정.")
            else:
                logging.error("RIDI 모드 OPF 수정 실패: manifest 또는 spine 찾을 수 없음.")

            ridi_opf_bytes = convert_opf_to_epub2_standard(opf_soup_copy, updated_metadata)
            if ridi_opf_bytes:
                final_opf_bytes = ridi_opf_bytes
                logging.info("RIDI 모드를 위해 OPF를 EPUB 2 표준 형식으로 변환했습니다.")
            else:
                logging.warning("OPF를 EPUB 2 표준으로 변환 실패. 수정된 OPF 구조 사용.")
                final_opf_bytes = finalize_opf(opf_soup_copy)
        else: # 'standard' 모드
            final_opf_bytes = finalize_opf(opf_soup_copy)

        if not final_opf_bytes:
            print_colored("오류: OPF 파일 최종 처리 실패.", colorama.Fore.RED)
            return None

        # CSS 처리
        global default_korean_style
        korean_style_content = load_korean_style(default_korean_style)
        linked_css_paths = identify_linked_css(original_content_map, original_zipinfo_map, json_data, opf_dir)
        modified_css_content_map = process_all_css(original_content_map, original_zipinfo_map, linked_css_paths, korean_style_content)

        # 최종 콘텐츠 맵 준비
        processed_content_map = {}
        mimetype_path = 'mimetype'
        if mimetype_path in original_content_map:
             processed_content_map[mimetype_path] = original_content_map[mimetype_path]
        else:
             logging.warning("원본 EPUB에 mimetype 파일이 없어 processed_content_map에 추가할 수 없습니다.")

        processed_content_map[opf_path] = final_opf_bytes
        processed_content_map.update(modified_css_content_map)

        logging.info("개별 콘텐츠 파일 처리 및 최종 내용 준비 중...")
        original_nav_doc_epub_path = None
        opf_manifest_check = opf_soup_copy.find('manifest')
        nav_item_check = opf_manifest_check.find('item', attrs={'properties': re.compile(r'\bnav\b')}) if opf_manifest_check else None

        if nav_item_check and nav_item_check.get('href'):
            nav_href_check = nav_item_check['href']
            original_nav_doc_epub_path = posixpath.normpath(posixpath.join(opf_dir, nav_href_check))
            logging.debug(f"원본 Nav Doc 경로 식별됨 (OPF 기준): {original_nav_doc_epub_path}")
        else:
            found_nav_by_name = False
            common_nav_names = ('nav.xhtml', 'navigation-documents.xhtml', 'toc.xhtml')
            for itempath_check in original_zipinfo_map.keys():
                 if posixpath.basename(itempath_check).lower() in common_nav_names:
                      original_nav_doc_epub_path = itempath_check
                      logging.debug(f"원본 Nav Doc 경로 식별됨 (파일명 기준 - 주의): {original_nav_doc_epub_path}")
                      found_nav_by_name = True
                      break
            if not found_nav_by_name:
                 logging.warning("OPF 또는 파일명 기준으로 Nav Doc 경로를 식별할 수 없습니다.")

        items_to_process = list(original_zipinfo_map.keys())

        if mode == 'ridi':
            if original_nav_doc_epub_path and original_nav_doc_epub_path in items_to_process:
                items_to_process.remove(original_nav_doc_epub_path)
                logging.debug(f"RIDI 모드: 처리 목록에서 원본 Nav Doc({original_nav_doc_epub_path}) 제거.")
            if ncx_target_path_in_epub and ncx_target_path_in_epub not in items_to_process:
                 items_to_process.append(ncx_target_path_in_epub)
                 logging.debug(f"RIDI 모드: 처리 목록에 NCX 경로({ncx_target_path_in_epub}) 추가 (또는 이미 존재).")

        for item_path in items_to_process:
            item_info = original_zipinfo_map.get(item_path)

            if item_path in processed_content_map: continue
            if item_info and item_info.is_dir():
                 logging.debug(f"Skipping directory entry: {item_path}")
                 continue

            file_basename = posixpath.basename(item_path)
            file_ext_lower = ''
            if '.' in file_basename: file_ext_lower = file_basename.lower().split('.')[-1]

            content_bytes = None
            is_processed = False
            is_toc_file = False
            toc_temp_path = None

            if mode == 'ridi' and item_path == ncx_target_path_in_epub:
                is_toc_file = True
                generated_ncx_temp_path = translated_toc_map.get('toc.ncx')
                original_ncx_temp_path = None
                for key, val in translated_toc_map.items():
                     if key.lower().endswith('.ncx') and key != 'toc.ncx':
                          original_ncx_temp_path = val
                          break

                if generated_ncx_temp_path and os.path.exists(generated_ncx_temp_path):
                    toc_temp_path = generated_ncx_temp_path
                    logging.debug(f"RIDI 모드: 생성된 NCX 파일 사용 ({os.path.basename(toc_temp_path)})")
                elif original_ncx_temp_path and os.path.exists(original_ncx_temp_path):
                    toc_temp_path = original_ncx_temp_path
                    logging.debug(f"RIDI 모드: 번역된 원본 NCX 파일 사용 ({os.path.basename(toc_temp_path)})")
                else:
                    logging.error(f"RIDI 모드 오류: NCX 파일({item_path})에 사용할 콘텐츠를 찾을 수 없습니다!")
                    content_bytes = b'<?xml version="1.0" encoding="UTF-8"?><ncx xmlns="http://www.daisy.org/z3986/2005/ncx/" version="2005-1"><head/><docTitle><text>Error: TOC Missing</text></docTitle><navMap/></ncx>'
                    is_processed = True

            elif mode == 'standard':
                if item_path in translated_toc_map:
                     toc_temp_path = translated_toc_map.get(item_path)
                     is_toc_file = True
                     logging.debug(f"표준 모드: 번역된 목차 파일 사용 ({os.path.basename(toc_temp_path)})")

            if is_toc_file and toc_temp_path and os.path.exists(toc_temp_path):
                 try:
                     with open(toc_temp_path, 'rb') as f_toc:
                         content_bytes = f_toc.read()
                     is_processed = True
                 except Exception as read_err:
                     logging.error(f"임시 목차 파일 읽기 실패 ({toc_temp_path}): {read_err}")
                     if mode == 'ridi':
                          content_bytes = b'<?xml version="1.0" encoding="UTF-8"?><ncx xmlns="http://www.daisy.org/z3986/2005/ncx/" version="2005-1"><head/><docTitle><text>Error: TOC Read Failed</text></docTitle><navMap/></ncx>'
                          is_processed = True
                     else:
                          content_bytes = original_content_map.get(item_path)
                          is_processed = False
            elif is_toc_file and mode == 'ridi' and content_bytes is None:
                 logging.error(f"RIDI 모드 오류: NCX 파일({item_path})에 사용할 콘텐츠 소스를 찾을 수 없습니다!")
                 content_bytes = b'<?xml version="1.0" encoding="UTF-8"?><ncx xmlns="http://www.daisy.org/z3986/2005/ncx/" version="2005-1"><head/><docTitle><text>Error: TOC Source Missing</text></docTitle><navMap/></ncx>'
                 is_processed = True

            elif file_ext_lower in ("xhtml", "html") and file_basename in json_data:
                logging.debug(f"처리 대상 XHTML: {item_path}")
                content_bytes = reconstruct_translated_xhtml(file_basename, json_data, output_dir, mode)
                if content_bytes is None:
                    logging.warning(f"XHTML 재구성 실패: {item_path}. 원본 사용 시도.")
                    content_bytes = original_content_map.get(item_path)
                else:
                    is_processed = True

            elif file_basename.lower().startswith("cover.") and file_ext_lower in ('jpg', 'jpeg', 'png', 'gif', 'webp'):
                 logging.debug(f"처리 대상 커버 이미지: {item_path}")
                 cover_original_bytes = original_content_map.get(item_path)
                 if cover_original_bytes:
                      content_bytes = process_cover_image(cover_original_bytes, cover_image_modify, cover_text_position, cover_text, font_path, font_size, font_color, background_color, script_dir)
                      if content_bytes != cover_original_bytes:
                           is_processed = True
                 else: logging.warning(f"커버 이미지 원본 내용을 찾을 수 없음: {item_path}")

            if content_bytes is None:
                 if item_path in original_content_map:
                     content_bytes = original_content_map.get(item_path)
                 else:
                     if mode == 'ridi' and item_path == ncx_target_path_in_epub:
                          logging.error(f"NCX 파일({item_path})의 내용을 찾거나 생성하지 못했습니다!")
                     else:
                          logging.warning(f"처리 대상 파일({item_path})의 내용을 원본에서도 찾을 수 없습니다. 건너뜁니다.")
                     continue

            if mode == 'ridi' and item_path.lower().endswith(('.xhtml', '.html')):
                logging.debug(f"RIDI 모드 헤더 적용 시도: {item_path}")
                try:
                    current_content_str = None
                    detected_encoding = 'utf-8'
                    try:
                        content_bytes_no_bom = content_bytes[3:] if content_bytes.startswith(b'\xef\xbb\xbf') else content_bytes
                        current_content_str = content_bytes_no_bom.decode(detected_encoding)
                    except UnicodeDecodeError:
                        try:
                            detected_encoding = 'cp949'
                            content_bytes_no_bom = content_bytes[3:] if content_bytes.startswith(b'\xef\xbb\xbf') else content_bytes
                            current_content_str = content_bytes_no_bom.decode(detected_encoding)
                            logging.info(f"RIDI 헤더 적용 위해 {item_path}를 {detected_encoding}으로 디코딩.")
                        except Exception as decode_err_inner:
                            logging.warning(f"RIDI 헤더 적용 위한 디코딩 실패 ({item_path}): {decode_err_inner}. 헤더 수정 건너뜁니다.")
                    except Exception as decode_err_outer:
                            logging.warning(f"RIDI 헤더 적용 위한 디코딩 중 오류 ({item_path}): {decode_err_outer}. 헤더 수정 건너뜁니다.")

                    if current_content_str is not None:
                        ridi_xml_declaration = '<?xml version="1.0" encoding="utf-8"?>\n'
                        ridi_doctype_declaration = '<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.1//EN"\n  "http://www.w3.org/TR/xhtml11/DTD/xhtml11.dtd">\n'
                        ridi_standard_header = ridi_xml_declaration + ridi_doctype_declaration

                        content_no_xml = re.sub(r"^\s*<\?xml[^>]*?\?>\s*", "", current_content_str, count=1, flags=re.IGNORECASE | re.DOTALL)
                        content_no_declarations = re.sub(r"^\s*<!DOCTYPE.*?>\s*", "", content_no_xml, count=1, flags=re.IGNORECASE | re.DOTALL)

                        final_ridi_content = ridi_standard_header + content_no_declarations.lstrip()

                        if final_ridi_content != current_content_str:
                             content_bytes = final_ridi_content.encode('utf-8')
                             logging.debug(f"    - {item_path}: RIDI 표준 헤더 적용/수정 완료.")

                except Exception as ridi_header_err:
                    logging.error(f"파일({item_path})에 RIDI 헤더 적용 중 오류: {ridi_header_err}", exc_info=True)

            processed_content_map[item_path] = content_bytes

        # --- (최종 EPUB 파일 경로 결정 및 write_epub_archive 호출 로직 유지) ---
        final_output_path = output_epub_path
        if final_output_path is None:
            epub_filename_base_only = os.path.splitext(json_data.get("epub_filename", "unknown_epub"))[0]

            json_filename_in_data = json_data.get("json_filename", "")
            is_second_pass = "_2nd trans" in json_filename_in_data if json_filename_in_data else False

            if is_second_pass:
                 suffix = "_ko_2nd trans_RIDI.epub" if mode == 'ridi' else "_ko_2nd trans.epub"
            else:
                 suffix = "_ko_RIDI.epub" if mode == 'ridi' else "_ko.epub"

            base_filename = f"{epub_filename_base_only}{suffix}"
            counter = 1
            final_output_path = os.path.join(epub_dir, base_filename)
            while os.path.exists(final_output_path):
                final_output_path = os.path.join(epub_dir, f"{epub_filename_base_only}{suffix.replace('.epub', '')} ({counter}).epub")
                counter += 1
        print_colored(f"EPUB 파일 저장 경로 ({mode} mode): {final_output_path}", colorama.Fore.WHITE)

        created_epub_path = write_epub_archive(
            final_output_path,
            original_content_map,
            original_zipinfo_map,
            processed_content_map,
            font_path,
            script_dir,
            opf_dir,
            mode=mode
        )
        return created_epub_path

    except Exception as e:
        print_colored(f"오류: EPUB 재구성 오케스트레이션 중 예외 발생: {e}", colorama.Fore.RED)
        logging.error(f"EPUB 재구성 오케스트레이션 중 예외 발생: {e}", exc_info=True)
        return None

        
def apply_regex_transformations(text):
    """텍스트에 정규식 변환을 적용합니다."""
    if not text:  # 빈 문자열 또는 None 처리
        return ""

    # '#### 번역 시작 #####' 이전 내용 삭제 (비탐욕적 매칭)
    text = re.sub(r".*?#+\s*번역\s*시작\s*#+", "", text, flags=re.DOTALL)
    
    # '############### 수정 시작 ###############' 이전 내용 삭제됨
    text = re.sub(r".*?#+\s*수정\s*시작\s*#+", "", text, flags=re.DOTALL)
    
    # 'current text:' 이전 내용 삭제 (비탐욕적 매칭)
    text = re.sub(r".*?current text:", "", text, flags=re.DOTALL)

    # 일본어 문장부호 변환
    text = text.replace("。", ".")
    text = text.replace("、", ",")
    text = text.replace("︑", ",")
    return text
    

def cleanup(output_dir):
    """임시 파일과 폴더를 삭제합니다."""
    deleted_successfully = False
    if os.path.exists(output_dir):
        try:
            shutil.rmtree(output_dir)  # 폴더와 하위 파일 모두 삭제
            deleted_successfully = True
        except OSError as e:
            print_colored(f"Error: 임시 폴더 삭제 실패: {e}", colorama.Fore.RED, colorama.Style.BRIGHT)
        finally:
            # 삭제 성공 시에만 완료 메시지 출력
            if deleted_successfully:
                 print_colored("임시 폴더 정리가 완료되었습니다.", colorama.Fore.CYAN)
            # 실패 메시지는 try 블록에서 이미 출력됨
    else:
         print_colored(f"Warning: 삭제할 임시 폴더({output_dir})를 찾을 수 없습니다.", colorama.Fore.YELLOW)
         
         
# --- 6. main 함수 ---        
if __name__ == "__main__":
    colorama.init()
    output_dir = None

    try:
        epub_file_path = input("EPUB 파일 경로를 입력하세요: ")
        epub_file_path = os.path.expandvars(epub_file_path.strip().strip('"'))
        epub_file_path = os.path.normpath(epub_file_path)
        epub_dir = os.path.dirname(epub_file_path)

        if os.path.exists(epub_file_path):
            epub_filename_base = os.path.splitext(os.path.basename(epub_file_path))[0]
            script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()

            (api_key, temperature, top_p, top_k, text_block_size, previous_context_number,
             retranslate_max_retries, second_translation, num_parallel,
             cover_image_modify, cover_text_position, cover_text, font_path, font_size,
             font_color, background_color, delete_temp_files,
             ridi_version) = get_api_key_and_params()

            updated_opf_soup, updated_metadata = get_and_update_metadata(epub_file_path)
            if updated_opf_soup is None or updated_metadata is None:
                print_colored("Error: 메타데이터 처리 중 오류 발생하여 프로그램을 종료합니다.", colorama.Fore.RED, colorama.Style.BRIGHT)
            else:
                selected_model = select_gemini_model(api_key)
                print_colored(f"선택된 모델: {selected_model}", colorama.Fore.WHITE, colorama.Style.BRIGHT)

                glossary_content = load_glossary()
                character_dictionary = load_character_dictionary()

                start_time = time.time()
                total_input_chars = 0
                total_output_chars = 0

                json_data, output_dir, failed_files, total_input_chars, total_output_chars = extract_epub_content(
                    epub_path=epub_file_path,
                    selected_model=selected_model,
                    api_key=api_key,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    glossary_content=glossary_content,
                    cover_image_modify=cover_image_modify,
                    cover_text_position=cover_text_position,
                    cover_text=cover_text,
                    font_path=font_path,
                    font_size=font_size,
                    font_color=font_color,
                    background_color=background_color,
                    retranslate_max_retries=retranslate_max_retries,
                    previous_context_number=previous_context_number,
                    text_block_size=text_block_size,
                    num_parallel=num_parallel,
                    character_dictionary=character_dictionary
                )

                final_toc_files_map = {}
                generated_ncx_from_nav_path = None # NCX 변환 결과 경로 초기화

                if json_data and output_dir:
                    try:
                        client = genai.Client(api_key=api_key)
                        model = GeminiModel(client, selected_model, safety_settings=safety_settings)
                    except Exception as model_init_err:
                         print_colored(f"Error: Gemini 모델 초기화 실패: {model_init_err}", colorama.Fore.RED)
                         model = None

                    if model:
                        print_colored("\n--- 목차 파일 처리 시작 ---", colorama.Fore.CYAN)
                        api_call_delay_for_toc = 0.6
                        ncx_info, nav_doc_info = identify_and_extract_toc_files(epub_file_path, output_dir)

                        translated_ncx_path = None
                        if ncx_info and 'temp_path' in ncx_info and 'original_filename' in ncx_info:
                            translated_ncx_path = translate_ncx_file(
                                ncx_temp_path=ncx_info['temp_path'],
                                output_dir=output_dir,
                                model=model,
                                updated_metadata=updated_metadata,
                                api_call_delay=api_call_delay_for_toc,
                                original_filename=ncx_info['original_filename']
                            )
                            if translated_ncx_path:
                                final_toc_files_map[ncx_info['epub_path']] = translated_ncx_path
                                print_colored(f"NCX 파일 번역 완료: {os.path.basename(translated_ncx_path)}", colorama.Fore.GREEN)
                            else:
                                final_toc_files_map[ncx_info['epub_path']] = ncx_info['temp_path']
                                print_colored(f"NCX 파일 번역 실패 또는 변경 없음. 원본 임시 파일({os.path.basename(ncx_info['temp_path'])}) 사용.", colorama.Fore.YELLOW)
                        elif ncx_info:
                             logging.warning(f"NCX 정보가 불완전하여 처리할 수 없습니다: {ncx_info}")

                        translated_nav_doc_path = None
                        if nav_doc_info and 'temp_path' in nav_doc_info and 'original_filename' in nav_doc_info:
                            translated_nav_doc_path = translate_nav_doc_file(
                                nav_doc_temp_path=nav_doc_info['temp_path'],
                                output_dir=output_dir,
                                model=model,
                                updated_metadata=updated_metadata,
                                api_call_delay=api_call_delay_for_toc,
                                original_filename=nav_doc_info['original_filename']
                            )
                            if translated_nav_doc_path:
                                final_toc_files_map[nav_doc_info['epub_path']] = translated_nav_doc_path
                                print_colored(f"Nav Doc 파일 번역 완료: {os.path.basename(translated_nav_doc_path)}", colorama.Fore.GREEN)

                                if ridi_version == 1:
                                    print_colored("RIDI 버전용 NCX 변환 시도 중...", colorama.Fore.CYAN)
                                    try:
                                        with open(translated_nav_doc_path, 'r', encoding='utf-8') as f_nav:
                                            nav_content_str = f_nav.read()
                                        ncx_bytes_generated = convert_nav_html_to_ncx(nav_content_str, updated_metadata)
                                        if ncx_bytes_generated:
                                            generated_ncx_from_nav_path = os.path.join(output_dir, "generated_toc.ncx")
                                            with open(generated_ncx_from_nav_path, 'wb') as f_ncx:
                                                f_ncx.write(ncx_bytes_generated)
                                            print_colored("RIDI 버전용 NCX 파일 생성 성공.", colorama.Fore.GREEN)
                                        else:
                                            print_colored("경고: RIDI 버전용 NCX 파일 변환 실패.", colorama.Fore.YELLOW)
                                    except Exception as conv_err:
                                        print_colored(f"오류: RIDI 버전용 NCX 변환 중 예외 발생: {conv_err}", colorama.Fore.RED)
                                        logging.error(f"NCX 변환 중 예외: {conv_err}", exc_info=True)
                            else:
                                final_toc_files_map[nav_doc_info['epub_path']] = nav_doc_info['temp_path']
                                print_colored(f"Nav Doc 파일 번역 실패 또는 변경 없음. 원본 임시 파일({os.path.basename(nav_doc_info['temp_path'])}) 사용.", colorama.Fore.YELLOW)
                        elif nav_doc_info:
                             logging.warning(f"Nav Doc 정보가 불완전하여 처리할 수 없습니다: {nav_doc_info}")

                        if not final_toc_files_map:
                             print_colored("경고: 처리할 목차 파일을 찾지 못했거나 처리 중 오류 발생.", colorama.Fore.YELLOW)
                        print_colored("--- 목차 파일 처리 완료 ---", colorama.Fore.CYAN)

                        returned_path_standard = None
                        returned_path_ridi = None

                        print_colored("\n--- 표준 번역 EPUB 생성 시작 ---", colorama.Fore.YELLOW)
                        returned_path_standard = rebuild_epub_orchestrator(
                            epub_path=epub_file_path,
                            json_data=json_data,
                            updated_opf_soup=updated_opf_soup,
                            updated_metadata=updated_metadata,
                            model=model,
                            cover_image_modify=cover_image_modify,
                            cover_text_position=cover_text_position,
                            cover_text=cover_text,
                            font_path=font_path,
                            font_size=font_size,
                            font_color=font_color,
                            background_color=background_color,
                            translated_toc_map=final_toc_files_map,
                            output_epub_path=None,
                            mode='standard'
                        )

                        if returned_path_standard:
                            print_colored("표준 번역 EPUB 생성 성공!", colorama.Fore.GREEN, colorama.Style.BRIGHT)

                            if ridi_version == 1:
                                print_colored("\n--- RIDI 버전 EPUB 생성 시작 ---", colorama.Fore.YELLOW)

                                final_toc_files_map_for_ridi = copy.deepcopy(final_toc_files_map)
                                if nav_doc_info and 'epub_path' in nav_doc_info and generated_ncx_from_nav_path:
                                    original_nav_epub_path = nav_doc_info['epub_path']
                                    if original_nav_epub_path in final_toc_files_map_for_ridi:
                                        del final_toc_files_map_for_ridi[original_nav_epub_path]
                                    final_toc_files_map_for_ridi['toc.ncx'] = generated_ncx_from_nav_path
                                    logging.info("RIDI EPUB용 목차 맵: Nav Doc 대신 생성된 NCX 사용.")
                                elif not ncx_info and not nav_doc_info:
                                     logging.warning("RIDI EPUB 생성: 원본 EPUB에 인식 가능한 목차 파일(NCX 또는 Nav Doc)이 없습니다.")
                                elif ncx_info:
                                     logging.info("RIDI EPUB용 목차 맵: 원본 NCX (번역본) 사용.")

                                ridi_suffix = "_ko_RIDI.epub"
                                ridi_base_filename = f"{epub_filename_base}{ridi_suffix}"
                                ridi_counter = 1
                                mod_epub_path_ridi = os.path.join(epub_dir, ridi_base_filename)
                                while os.path.exists(mod_epub_path_ridi):
                                    mod_epub_path_ridi = os.path.join(epub_dir, f"{epub_filename_base}{ridi_suffix.replace('.epub', '')} ({ridi_counter}).epub")
                                    ridi_counter += 1

                                returned_path_ridi = rebuild_epub_orchestrator(
                                    epub_path=epub_file_path,
                                    json_data=json_data,
                                    updated_opf_soup=copy.deepcopy(updated_opf_soup),
                                    updated_metadata=copy.deepcopy(updated_metadata),
                                    model=model,
                                    cover_image_modify=cover_image_modify,
                                    cover_text_position=cover_text_position,
                                    cover_text=cover_text,
                                    font_path=font_path,
                                    font_size=font_size,
                                    font_color=font_color,
                                    background_color=background_color,
                                    translated_toc_map=final_toc_files_map_for_ridi,
                                    output_epub_path=mod_epub_path_ridi,
                                    mode='ridi'
                                )

                                if returned_path_ridi:
                                    print_colored("RIDI 버전 EPUB 생성 성공!", colorama.Fore.GREEN, colorama.Style.BRIGHT)
                                else:
                                    print_colored("Error: RIDI 버전 EPUB 파일 생성에 실패했습니다.", colorama.Fore.RED, colorama.Style.BRIGHT)
                            else:
                                print_colored("설정에 따라 RIDI 버전 생성을 건너뜁니다.", colorama.Fore.YELLOW, colorama.Style.BRIGHT)
                        else:
                            print_colored("Error: 표준 번역 EPUB 파일 생성에 실패하여 RIDI 버전 생성을 건너뜁니다.", colorama.Fore.RED, colorama.Style.BRIGHT)

                        if returned_path_standard and second_translation != 1:
                            end_time = time.time()
                            elapsed_time = end_time - start_time
                            estimated_total_tokens = total_input_chars * 0.6 + total_output_chars * 0.75
                            

                            print(f"\n--- 1차 번역 결과 ---")
                            print(f"총 번역 시간: {str(datetime.timedelta(seconds=int(elapsed_time)))}")
                            print(f"총 사용 글자 수 (Input + Output): {int(total_input_chars + total_output_chars)} 글자")

                        json_data_2nd = None
                        returned_path_standard_2nd = None

                        if returned_path_standard and second_translation == 1:
                            print_colored("\n--- 2차 번역 수행 시작 ---", colorama.Fore.YELLOW)

                            json_data_2nd, json_2nd_filename, returned_path_standard_2nd = perform_second_translation(
                                output_dir=output_dir,
                                selected_model=selected_model,
                                api_key=api_key,
                                temperature=temperature,
                                top_p=top_p,
                                top_k=top_k,
                                glossary_content=glossary_content,
                                character_dictionary=character_dictionary,
                                json_data=json_data,
                                epub_file_path=epub_file_path,
                                epub_dir=epub_dir,
                                updated_opf_soup=updated_opf_soup,
                                updated_metadata=updated_metadata,
                                model=model,
                                cover_image_modify=cover_image_modify,
                                cover_text_position=cover_text_position,
                                cover_text=cover_text,
                                font_path=font_path,
                                font_size=font_size,
                                font_color=font_color,
                                background_color=background_color,
                                total_input_chars_1st=total_input_chars,
                                total_output_chars_1st=total_output_chars,
                                start_time=start_time,
                                translated_toc_map=final_toc_files_map,
                                num_parallel=num_parallel,
                                ridi_version=ridi_version
                            )

                        elif not returned_path_standard:
                             print_colored("1차 표준 EPUB 생성 실패로 2차 번역을 건너뜁니다.", colorama.Fore.YELLOW, colorama.Style.BRIGHT)
                        elif second_translation != 1:
                             print_colored("2차 번역 설정을 사용하지 않아 건너뜁니다.", colorama.Fore.YELLOW, colorama.Style.BRIGHT)

                    else:
                         print_colored("Error: 모델 초기화 실패로 EPUB 생성을 진행할 수 없습니다.", colorama.Fore.RED)

                elif not json_data:
                    print_colored("Error: EPUB 내용 추출 또는 초기 번역 단계에서 실패했습니다.", colorama.Fore.RED, colorama.Style.BRIGHT)
                elif not output_dir:
                    print_colored("Error: 임시 작업 디렉토리 경로를 결정할 수 없습니다.", colorama.Fore.RED)

        else:
            print_colored("Error: EPUB 파일을 찾을 수 없습니다.", colorama.Fore.RED, colorama.Style.BRIGHT)

    except Exception as e:
        print_colored(f"\n스크립트 실행 중 예외 발생: {e}", colorama.Fore.RED, colorama.Style.BRIGHT)
        import traceback
        traceback.print_exc()

    finally:
        try:
            root_logger = logging.getLogger()
            if root_logger.hasHandlers():
                for handler in root_logger.handlers[:]:
                    try:
                        handler.close()
                        root_logger.removeHandler(handler)
                    except Exception as h_close_err:
                        print_colored(f"Warning: 로깅 핸들러 닫기 중 오류: {h_close_err}", colorama.Fore.YELLOW)
            else:
                pass
        except Exception as log_e:
            print_colored(f"Warning: 로깅 종료 중 오류 발생: {log_e}", colorama.Fore.YELLOW)

        if 'delete_temp_files' in locals() and delete_temp_files == 1:
            if output_dir and os.path.exists(output_dir):
                print_colored(f"\n임시 폴더({output_dir}) 정리 중...", colorama.Fore.CYAN)
                cleanup(output_dir)
            else:
                 print_colored("\n삭제할 임시 폴더가 없거나 경로가 유효하지 않습니다.", colorama.Fore.YELLOW)
        elif 'delete_temp_files' in locals() and delete_temp_files != 1:
            print_colored("\n설정에 따라 임시 파일을 삭제하지 않습니다.", colorama.Fore.YELLOW)
        elif 'delete_temp_files' not in locals():
             print_colored("\n임시 파일 삭제 설정 값을 결정하지 못했습니다.", colorama.Fore.YELLOW)


        print("\n=========================================")
        input("모든 작업이 완료되었습니다. 아무 키나 눌러 창을 닫으세요...")
        colorama.deinit()