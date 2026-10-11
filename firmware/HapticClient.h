#pragma once
// V2 strengths [Left, Center, Right], 0..100. ArduinoJson 7 / ESP32 Arduino 3.x.
#include <Arduino.h>
#include <ArduinoJson.h>
#include <HTTPClient.h>
#include <WiFi.h>
#include <freertos/FreeRTOS.h>
#include <freertos/semphr.h>
#include <freertos/task.h>

class HapticClient {
 public:
  // URL and optional token must outlive this object. Use string literals.
  explicit HapticClient(const char* url, const char* token = nullptr)
      : url_(url), token_(token) {}

  bool begin() {
    if (task_) return true;
    if (!mutex_) mutex_ = xSemaphoreCreateMutex();
    return mutex_ && xTaskCreate(taskEntry, "haptic-http", 8192, this, 1, &task_) == pdPASS;
  }

  bool read(int strengths[3]) {
    for (int i = 0; i < 3; ++i) strengths[i] = 0;
    if (!mutex_) return false;
    xSemaphoreTake(mutex_, portMAX_DELAY);
    bool fresh = valid_ && WiFi.status() == WL_CONNECTED && static_cast<int32_t>(deadline_ - millis()) > 0;
    if (fresh) for (int i = 0; i < 3; ++i) strengths[i] = strengths_[i];
    xSemaphoreGive(mutex_);
    return fresh;
  }

 private:
  void invalidate() {
    xSemaphoreTake(mutex_, portMAX_DELAY);
    valid_ = false;
    xSemaphoreGive(mutex_);
  }

  bool poll() {
    if (WiFi.status() != WL_CONNECTED) return false;
    const uint32_t started = millis();
    WiFiClient socket;
    HTTPClient http;
    http.setConnectTimeout(700);
    http.setTimeout(700);
    http.useHTTP10(true);
    if (!http.begin(socket, url_)) return false;
    if (token_ && token_[0]) http.addHeader("Authorization", String("Bearer ") + token_);
    const int status = http.GET();
    if (status != 200 || http.getSize() <= 0 || http.getSize() > 4096) {
      http.end();
      return false;
    }
    const String payload = http.getString();
    http.end();
    JsonDocument doc;
    if (deserializeJson(doc, payload)) return false;
    if (!doc["schema_version"].is<int>() || doc["schema_version"].as<int>() != 2 ||
        !doc["valid"].is<bool>() || !doc["valid"].as<bool>() ||
        !doc["command_id"].is<uint32_t>() || !doc["session_id"].is<const char*>() ||
        !doc["ttl_ms"].is<uint32_t>()) return false;
    const uint32_t ttl = doc["ttl_ms"].as<uint32_t>();
    if (!ttl || ttl > 2500 || millis() - started >= ttl) return false;
    const String session = doc["session_id"].as<String>();
    if (session.length() != 32) return false;
    for (unsigned i = 0; i < 32; ++i) {
      char c = session[i];
      if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'))) return false;
    }
    const uint32_t command = doc["command_id"].as<uint32_t>();
    JsonArrayConst levels = doc["motor_strength_3"].as<JsonArrayConst>();
    JsonArrayConst available = doc["motor_valid_3"].as<JsonArrayConst>();
    if (levels.size() != 3 || available.size() != 3) return false;
    int next[3];
    for (int i = 0; i < 3; ++i) {
      if (!levels[i].is<int>() || !available[i].is<bool>()) return false;
      next[i] = levels[i].as<int>();
      if (next[i] < 0 || next[i] > 100) return false;
      if (!available[i].as<bool>()) next[i] = 0;
    }
    const uint32_t candidateDeadline = started + ttl;
    const bool same = haveCommand_ && session == session_ && command == command_;
    if (haveCommand_ && session == session_ && command < command_) return false;
    xSemaphoreTake(mutex_, portMAX_DELAY);
    if (same) {
      // Polling cannot extend the lease, change output, or revive invalidated output.
      if (static_cast<int32_t>(candidateDeadline - deadline_) < 0) deadline_ = candidateDeadline;
    } else {
      for (int i = 0; i < 3; ++i) strengths_[i] = next[i];
      deadline_ = candidateDeadline;
      valid_ = static_cast<int32_t>(deadline_ - millis()) > 0;
    }
    const bool accepted = valid_ && static_cast<int32_t>(deadline_ - millis()) > 0;
    xSemaphoreGive(mutex_);
    session_ = session;
    command_ = command;
    haveCommand_ = true;
    if (accepted && !same) Serial.printf("Motor strength %% [L,C,R]: [%d,%d,%d]\n", next[0], next[1], next[2]);
    return accepted;
  }

  static void taskEntry(void* pointer) {
    auto* self = static_cast<HapticClient*>(pointer);
    for (;;) {
      if (!self->poll()) self->invalidate();
      vTaskDelay(pdMS_TO_TICKS(100));
    }
  }

  const char* url_;
  const char* token_;
  SemaphoreHandle_t mutex_ = nullptr;
  TaskHandle_t task_ = nullptr;
  int strengths_[3] = {0, 0, 0};
  bool valid_ = false;
  bool haveCommand_ = false;
  String session_;
  uint32_t command_ = 0;
  uint32_t deadline_ = 0;
};
