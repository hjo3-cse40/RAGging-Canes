#pragma once

// Add to the existing Arduino ESP32 camera sketch. Requires ArduinoJson 7.
// Receives and prints zones only: no GPIO writes or motor control.
#include <Arduino.h>
#include <ArduinoJson.h>
#include <HTTPClient.h>
#include <WiFi.h>
#include <freertos/FreeRTOS.h>
#include <freertos/semphr.h>
#include <freertos/task.h>

class SixZoneClient {
 public:
  explicit SixZoneClient(const char* endpoint, const char* token = nullptr)
      : endpoint_(endpoint), token_(token) {}

  bool begin() {
    if (task_ != nullptr) return true;
    if (mutex_ == nullptr) mutex_ = xSemaphoreCreateMutex();
    if (mutex_ == nullptr) return false;
    return xTaskCreate(runTask, "zone-poll", 8192, this, 1, &task_) == pdPASS;
  }

  // Returns false and fills -1 when unavailable/expired. Call from loop() if needed.
  bool read(int output[6]) {
    if (mutex_ == nullptr) {
      for (int i = 0; i < 6; ++i) output[i] = -1;
      return false;
    }
    xSemaphoreTake(mutex_, portMAX_DELAY);
    bool fresh = valid_ && static_cast<int32_t>(deadline_ - millis()) > 0;
    for (int i = 0; i < 6; ++i) output[i] = fresh ? zones_[i] : -1;
    xSemaphoreGive(mutex_);
    return fresh;
  }

 private:
  const char* endpoint_;  // Keep the supplied string alive (use a string literal).
  const char* token_;     // Keep alive; store credentials outside committed code.
  SemaphoreHandle_t mutex_ = nullptr;
  TaskHandle_t task_ = nullptr;
  int zones_[6] = {-1, -1, -1, -1, -1, -1};
  bool valid_ = false;
  uint32_t deadline_ = 0;
  uint32_t frame_ = 0;
  String session_;
  bool announcedUnknown_ = false;

  void invalidate() {
    xSemaphoreTake(mutex_, portMAX_DELAY);
    valid_ = false;
    xSemaphoreGive(mutex_);
    if (!announcedUnknown_) Serial.println("Zones unavailable/expired: [-1,-1,-1,-1,-1,-1]");
    announcedUnknown_ = true;
  }

  bool poll() {
    if (WiFi.status() != WL_CONNECTED) return false;
    uint32_t requestedAt = millis();
    WiFiClient socket;
    HTTPClient http;
    http.setConnectTimeout(700);
    http.setTimeout(700);
    http.useHTTP10(true);
    if (!http.begin(socket, endpoint_)) return false;
    if (token_ != nullptr) http.addHeader("Authorization", String("Bearer ") + token_);
    int status = http.GET();
    if (status != HTTP_CODE_OK || http.getSize() <= 0 || http.getSize() > 4096) {
      http.end();
      return false;
    }
    String payload = http.getString();
    http.end();
    JsonDocument doc;
    if (deserializeJson(doc, payload)) return false;
    if (!doc["valid"].is<bool>() || !doc["valid"].as<bool>() ||
        !doc["ttl_ms"].is<int>() || !doc["frame_id"].is<uint32_t>() ||
        !doc["session_id"].is<const char*>()) return false;
    JsonArrayConst values = doc["zones"].as<JsonArrayConst>();
    if (values.size() != 6) return false;
    int next[6];
    for (int i = 0; i < 6; ++i) {
      if (!values[i].is<int>()) return false;
      next[i] = values[i].as<int>();
      if (next[i] < -1 || next[i] > 1) return false;
    }
    int ttl = doc["ttl_ms"].as<int>();
    // Start TTL at request time, conservatively including the entire HTTP delay.
    if (ttl <= 0 || ttl > 2500 || millis() - requestedAt >= static_cast<uint32_t>(ttl)) return false;
    uint32_t frame = doc["frame_id"].as<uint32_t>();
    String session = doc["session_id"].as<String>();
    if (session.length() != 32) return false;
    uint32_t deadline = requestedAt + static_cast<uint32_t>(ttl);
    bool changed = session != session_ || frame != frame_;
    if (session == session_ && frame < frame_) return false;
    xSemaphoreTake(mutex_, portMAX_DELAY);
    // Re-polling the same frame must never extend its original local deadline.
    if (!changed && static_cast<int32_t>(deadline - deadline_) > 0) deadline = deadline_;
    for (int i = 0; i < 6; ++i) zones_[i] = next[i];
    deadline_ = deadline;
    valid_ = static_cast<int32_t>(deadline_ - millis()) > 0;
    bool accepted = valid_;
    xSemaphoreGive(mutex_);
    session_ = session;
    frame_ = frame;
    if (!accepted) return false;
    if (changed || announcedUnknown_) {
      Serial.printf("Frame %lu zones: [%d,%d,%d,%d,%d,%d]\n",
                    static_cast<unsigned long>(frame), next[0], next[1], next[2], next[3], next[4], next[5]);
    }
    announcedUnknown_ = false;
    return true;
  }

  static void runTask(void* pointer) {
    SixZoneClient* self = static_cast<SixZoneClient*>(pointer);
    for (;;) {
      if (!self->poll()) self->invalidate();
      vTaskDelay(pdMS_TO_TICKS(200));
    }
  }
};
