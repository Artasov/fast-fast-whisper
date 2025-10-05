#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Тестовый скрипт для проверки параметра device
"""
import requests
import os

def test_device_parameter():
    """Тестирует параметр device"""
    url = "http://localhost:8000/v1/audio/transcriptions"
    
    # Проверяем, что тестовый файл существует
    test_file = "test_recognize/test.wav"
    if not os.path.exists(test_file):
        print(f"ОШИБКА: Тестовый файл {test_file} не найден")
        return
    
    print("Тестирование параметра device...")
    
    # Тест 1: Принудительное использование CPU
    print("\n1. Тестируем device=cpu...")
    try:
        with open(test_file, 'rb') as f:
            files = {'file': f}
            data = {
                'model': 'tiny',
                'response_format': 'json',
                'device': 'cpu'
            }
            response = requests.post(url, files=files, data=data)
            
        if response.status_code == 200:
            result = response.json()
            print(f"УСПЕХ CPU тест: {result.get('text', '')[:50]}...")
        else:
            print(f"ОШИБКА CPU тест: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"ОШИБКА при тестировании CPU: {e}")
    
    # Тест 2: Принудительное использование CUDA (если доступно)
    print("\n2. Тестируем device=cuda...")
    try:
        with open(test_file, 'rb') as f:
            files = {'file': f}
            data = {
                'model': 'tiny',
                'response_format': 'json',
                'device': 'cuda'
            }
            response = requests.post(url, files=files, data=data)
            
        if response.status_code == 200:
            result = response.json()
            print(f"УСПЕХ CUDA тест: {result.get('text', '')[:50]}...")
        else:
            print(f"ОШИБКА CUDA тест: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"ОШИБКА при тестировании CUDA: {e}")
    
    # Тест 3: Автоматический выбор (auto)
    print("\n3. Тестируем device=auto...")
    try:
        with open(test_file, 'rb') as f:
            files = {'file': f}
            data = {
                'model': 'tiny',
                'response_format': 'json',
                'device': 'auto'
            }
            response = requests.post(url, files=files, data=data)
            
        if response.status_code == 200:
            result = response.json()
            print(f"УСПЕХ AUTO тест: {result.get('text', '')[:50]}...")
        else:
            print(f"ОШИБКА AUTO тест: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"ОШИБКА при тестировании AUTO: {e}")
    
    # Тест 4: Неверный параметр device
    print("\n4. Тестируем неверный device...")
    try:
        with open(test_file, 'rb') as f:
            files = {'file': f}
            data = {
                'model': 'tiny',
                'response_format': 'json',
                'device': 'invalid_device'
            }
            response = requests.post(url, files=files, data=data)
            
        if response.status_code == 400:
            print("УСПЕХ: Валидация работает - неверный device отклонен")
        else:
            print(f"ОШИБКА: Валидация не работает: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"ОШИБКА при тестировании валидации: {e}")

if __name__ == "__main__":
    test_device_parameter()
