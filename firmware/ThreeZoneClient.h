#pragma once
// Three-zone [Left, Center, Right] polling client for the 3-motor wearable MVP.
// Polls `/command?zones=3` and validates TTL/freshness without driving GPIO directly.

#include <Arduino.h>
#include <ArduinoJson.h>
#include <HTTPClient.h>
#include <WiFi.h>

class ThreeZoneClient {
 public:
  explicit ThreeZoneClient(const char* commandUrl, const char* bearerToken = nullptr,
                           uint32_t pollIntervalMs = 80, uint32_t httpTimeoutMs = 600)
      : commandUrl_(commandUrl),
        bearerToken_(bearerToken),
        pollIntervalMs_(pollIntervalMs),
        httpTimeoutMs_(httpTimeoutMs),
        mutex_(xSemaphoreCreateMutex()) {}

  bool begin() {
    if (!mutex_ || taskHandle_ != nullptr) return false;
    return xTaskCreatePinnedToCore(taskEntry, "ThreeZoneClient", 6144, this, 1, &taskHandle_, 0) == pdPASS;
  }

  bool read(int outZones[3]) {
    if (!mutex_) {
      for (int i = 0; i < 3; ++i) outZones[i] = -1;
      return false;
    }
    xSemaphoreTake(mutex_, portMAX_DELAY);
    const uint32_t now = millis();
    const bool fresh = valid_ && static_cast<int32_t>(expiresAtMs_ - now) > 0;
    for (int i = 0; i < 3; ++i) {
      outZones[i] = fresh ? zones_[i] : -1;
    }
    xSemaphoreGive(mutex_);
    return fresh;
  }

 private:
  static void taskEntry(void* arg) {
    static_cast<ThreeZoneClient*>(arg)->run();
  }

  void invalidate(const char* reason) {
    xSemaphoreTake(mutex_, portMAX_DELAY);
    const bool wasValid = valid_;
    valid_ = false;
    for (int i = 0; i < 3; ++i) zones_[i] = -1;
    xSemaphoreGive(mutex_);
    if (wasValid || !printedUnavailable_) {
      Serial.printf("3-Zones unavailable/expired (%s): [-1,-1,-1]\n", reason);
      printedUnavailable_ = true;
    }
  }

  void run() {
    WiFiClient client;
    HTTPClient http;
    http.setReuse(true);
    for (;;) {
      if (WiFi.status() != WL_CONNECTED) {
        invalidate("wifi");
        vTaskDelay(pdMS_TO_TICKS(pollIntervalMs_));
        continue;
      }
      client.setNoDelay(true);
      const uint32_t started = millis();
      http.setTimeout(httpTimeoutMs_);
      if (!http.begin(client, commandUrl_)) {
        invalidate("begin");
        vTaskDelay(pdMS_TO_TICKS(pollIntervalMs_));
        continue;
      }
      if (bearerToken_ && bearerToken_[0] != '\0') {
        http.addHeader("Authorization", String("Bearer ") + bearerToken_);
      }
      const int status = http.GET();
      const String payload = status == 200 ? http.getString() : String();
      const uint32_t rtt = millis() - started;

      if (status != 200) {
        http.end();
        invalidate("http");
        vTaskDelay(pdMS_TO_TICKS(pollIntervalMs_));
        continue;
      }

      JsonDocument doc;
      if (deserializeJson(doc, payload) != DeserializationError::Ok) {
        invalidate("json");
        vTaskDelay(pdMS_TO_TICKS(pollIntervalMs_));
        continue;
      }

      const bool valid = doc["valid"].is<bool>() && doc["valid"].as<bool>();
      const uint32_t ttl = doc["ttl_ms"].is<uint32_t>() ? doc["ttl_ms"].as<uint32_t>() : 0;
      const uint32_t frameId = doc["frame_id"].is<uint32_t>() ? doc["frame_id"].as<uint32_t>() : 0;
      const char* session = doc["session_id"].is<const char*>() ? doc["session_id"].as<const char*>() : "";
      JsonArray arr = doc["zones_3"].is<JsonArray>() ? doc["zones_3"].as<JsonArray>() : doc["zones"].as<JsonArray>();

      if (!valid || arr.isNull() || arr.size() != 3 || ttl == 0 || ttl > 2500 || rtt >= ttl) {
        invalidate("stale_or_schema");
        vTaskDelay(pdMS_TO_TICKS(pollIntervalMs_));
        continue;
      }

      int parsed[3];
      bool ok = true;
      for (int i = 0; i < 3; ++i) {
        if (!arr[i].is<int>()) { ok = false; break; }
        parsed[i] = arr[i].as<int>();
        if (parsed[i] != 0 && parsed[i] != 1) { ok = false; break; }
      }
      if (!ok) {
        invalidate("values");
        vTaskDelay(pdMS_TO_TICKS(pollIntervalMs_));
        continue;
      }

      const uint32_t candidateDeadline = millis() + (ttl - rtt);
      xSemaphoreTake(mutex_, portMAX_DELAY);
      const bool sameFrame = (lastSession_ == session && lastFrameId_ == frameId);
      if (!sameFrame) {
        lastSession_ = session;
        lastFrameId_ = frameId;
        expiresAtMs_ = candidateDeadline;
        for (int i = 0; i < 3; ++i) zones_[i] = parsed[i];
        valid_ = true;
        printedUnavailable_ = false;
        xSemaphoreGive(mutex_);
        Serial.printf("Frame %u 3-zones [L,C,R]: [%d,%d,%d]\n",
                      frameId, parsed[0], parsed[1], parsed[2]);
      } else {
        if (static_cast<int32_t>(candidateDeadline - expiresAtMs_) < 0) {
          expiresAtMs_ = candidateDeadline;
        }
        xSemaphoreGive(mutex_);
      }
      vTaskDelay(pdMS_TO_TICKS(pollIntervalMs_));
    }
  }

  const char* commandUrl_;
  const char* bearerToken_;
  uint32_t pollIntervalMs_;
  uint32_t httpTimeoutMs_;
  SemaphoreHandle_t mutex_;
  TaskHandle_t taskHandle_ = nullptr;
  String lastSession_;
  uint32_t lastFrameId_ = 0;
  uint32_t expiresAtMs_ = 0;
  int zones_[3] = {-1, -1, -1};
  bool valid_ = false;
  bool printedUnavailable_ = false;
};
