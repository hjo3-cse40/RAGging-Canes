#pragma once
// FastStreamServer.h — Low-latency 320x240 (QVGA) persistent MJPEG & WebSocket
// camera server for Freenove ESP32-S3 WROOM / ESP32-CAM.
//
// Targets ~45–85 ms frame delivery (~12–20 FPS) over Wi-Fi by combining:
//   1. FRAMESIZE_QVGA (320x240) + JPEG quality 14 (~4–7 KB/frame vs ~35 KB VGA)
//   2. CAMERA_GRAB_LATEST with 2 PSRAM framebuffers (never queues stale frames)
//   3. WiFi.setSleep(false) / WIFI_PS_NONE (eliminates 100–300 ms DTIM sleep spikes)
//   4. Persistent HTTP multipart MJPEG (/stream) and binary WebSocket (/ws)
//      with TCP_NODELAY enabled on the underlying lwIP socket.

#include <Arduino.h>
#include <WiFi.h>
#include "esp_camera.h"
#include "esp_http_server.h"
#include "esp_timer.h"
#include "esp_wifi.h"
#include "lwip/sockets.h"
#include "mbedtls/base64.h"
#include "mbedtls/sha1.h"

class FastStreamServer {
 public:
  explicit FastStreamServer(uint16_t port = 80, uint16_t streamPort = 81)
      : port_(port), streamPort_(streamPort) {}

  // Call before esp_camera_init(&config) to configure dual-buffer latest-grab QVGA.
  static void applyLowLatencyConfig(camera_config_t& config,
                                    framesize_t frameSize = FRAMESIZE_QVGA,
                                    int jpegQuality = 14) {
    config.pixel_format = PIXFORMAT_JPEG;
    config.frame_size = frameSize;
    config.jpeg_quality = jpegQuality;
    config.fb_count = psramFound() ? 2 : 1;
    config.fb_location = psramFound() ? CAMERA_FB_IN_PSRAM : CAMERA_FB_IN_DRAM;
    config.grab_mode = psramFound() ? CAMERA_GRAB_LATEST : CAMERA_GRAB_WHEN_EMPTY;
  }

  // Call in setup() after WiFi.begin() and esp_camera_init() to switch even an
  // existing sketch to 320x240 (FRAMESIZE_QVGA) and disable Wi-Fi modem sleep.
  static bool optimizeLiveSensor(framesize_t frameSize = FRAMESIZE_QVGA,
                                 int jpegQuality = 14) {
    WiFi.setSleep(false);
    esp_wifi_set_ps(WIFI_PS_NONE);
    sensor_t* s = esp_camera_sensor_get();
    if (!s) return false;
    if (s->pixformat == PIXFORMAT_JPEG) {
      s->set_framesize(s, frameSize);
      s->set_quality(s, jpegQuality);
    }
    return true;
  }

  // Configure 3 directional vibration motor PWM pins [Left, Center, Right].
  // Defaults match hardware wiring: LEFT=42, CENTER=21, RIGHT=47, 1000 Hz, 8-bit PWM (0..255).
  bool configureMotors(int pinLeft = 42, int pinCenter = 21, int pinRight = 47,
                       uint32_t pwmFreq = 1000, uint8_t pwmResolution = 8) {
    motorPins_[0] = pinLeft;
    motorPins_[1] = pinCenter;
    motorPins_[2] = pinRight;
    pwmFreq_ = pwmFreq;
    pwmResolution_ = pwmResolution;
    bool allOk = true;
    for (int i = 0; i < 3; ++i) {
      if (motorPins_[i] >= 0) {
        if (ledcAttach(motorPins_[i], pwmFreq_, pwmResolution_)) {
          ledcWrite(motorPins_[i], 0);
          motorPwmAttached_[i] = true;
        } else {
          pinMode(motorPins_[i], OUTPUT);
          digitalWrite(motorPins_[i], LOW);
          motorPwmAttached_[i] = false;
          allOk = false;
        }
      }
    }
    ensureSensorTask();
    return allOk;
  }

  static int getPWM(int strength) {
    return map(constrain(strength, 0, 100), 0, 100, 0, 255);
  }

  // Optional: configure HC-SR04 / ultrasonic proximity sensor pins and local center-motor reflex.
  void configureUltrasonic(int trigPin, int echoPin, float reflexThresholdCm = 60.0f) {
    ultraTrigPin_ = trigPin;
    ultraEchoPin_ = echoPin;
    ultraReflexCm_ = reflexThresholdCm;
    if (ultraTrigPin_ >= 0 && ultraEchoPin_ >= 0) {
      pinMode(ultraTrigPin_, OUTPUT);
      digitalWrite(ultraTrigPin_, LOW);
      pinMode(ultraEchoPin_, INPUT);
    }
    ensureSensorTask();
  }

  // Read latest ultrasonic distance in cm (-1.0f if unavailable).
  float readUltrasonicCm() const { return lastUltrasonicCm_; }

  // Read current applied motor strengths [Left, Center, Right] (0..100).
  void readMotorStrengths(int outStrengths[3]) const {
    for (int i = 0; i < 3; ++i) outStrengths[i] = appliedStrengths_[i];
  }

  bool begin(framesize_t initialFrameSize = FRAMESIZE_QVGA, int initialQuality = 14) {
    optimizeLiveSensor(initialFrameSize, initialQuality);
    ensureSensorTask();

    httpd_config_t config = HTTPD_DEFAULT_CONFIG();
    config.server_port = port_;
    config.ctrl_port = 32768 + (port_ % 1000);
    config.max_uri_handlers = 10;
    config.max_open_sockets = 5;
    config.lru_purge_enable = true;

    if (httpd_start(&cmdServer_, &config) != ESP_OK) {
      return false;
    }

    const httpd_uri_t captureUri = {"/capture", HTTP_GET, handleCapture, this};
    const httpd_uri_t controlUri = {"/control", HTTP_GET, handleControl, this};
    const httpd_uri_t statusUri = {"/status", HTTP_GET, handleStatus, this};
    const httpd_uri_t hapticsUri = {"/haptics", HTTP_GET, handleHaptics, this};
    const httpd_uri_t streamUri = {"/stream", HTTP_GET, handleStream, this};
    const httpd_uri_t wsUri = {"/ws", HTTP_GET, handleWebSocket, this};

    httpd_register_uri_handler(cmdServer_, &captureUri);
    httpd_register_uri_handler(cmdServer_, &controlUri);
    httpd_register_uri_handler(cmdServer_, &statusUri);
    httpd_register_uri_handler(cmdServer_, &hapticsUri);
    // Expose /stream, /ws, and /haptics on the primary port (e.g. port 80) so single-port
    // SSH reverse tunnels (-R 8117:ESP32_IP:80) carry video, ultrasonic, and haptics together.
    httpd_register_uri_handler(cmdServer_, &streamUri);
    httpd_register_uri_handler(cmdServer_, &wsUri);

    if (streamPort_ != 0 && streamPort_ != port_) {
      httpd_config_t streamConfig = HTTPD_DEFAULT_CONFIG();
      streamConfig.server_port = streamPort_;
      streamConfig.ctrl_port = config.ctrl_port + 1;
      streamConfig.max_uri_handlers = 4;
      streamConfig.max_open_sockets = 3;
      streamConfig.lru_purge_enable = true;
      if (httpd_start(&streamServer_, &streamConfig) == ESP_OK) {
        httpd_register_uri_handler(streamServer_, &streamUri);
        httpd_register_uri_handler(streamServer_, &wsUri);
      }
    }
    return true;
  }

 private:
  static void enableTcpNoDelay(httpd_req_t* req) {
    const int fd = httpd_req_to_sockfd(req);
    if (fd >= 0) {
      int nodelay = 1;
      setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &nodelay, sizeof(nodelay));
    }
  }

  static void applyQuerySensorParams(httpd_req_t* req) {
    char query[96];
    if (httpd_req_get_url_query_str(req, query, sizeof(query)) != ESP_OK) return;
    sensor_t* s = esp_camera_sensor_get();
    if (!s) return;
    char valBuf[16];
    if (httpd_query_key_value(query, "framesize", valBuf, sizeof(valBuf)) == ESP_OK ||
        httpd_query_key_value(query, "size", valBuf, sizeof(valBuf)) == ESP_OK) {
      int fs = parseFrameSize(valBuf);
      if (fs >= 0) s->set_framesize(s, static_cast<framesize_t>(fs));
    }
    if (httpd_query_key_value(query, "quality", valBuf, sizeof(valBuf)) == ESP_OK) {
      int q = atoi(valBuf);
      if (q >= 4 && q <= 63) s->set_quality(s, q);
    }
  }

  static int parseFrameSize(const char* val) {
    if (!val || !val[0]) return -1;
    if (strcasecmp(val, "qqvga") == 0 || strcasecmp(val, "160x120") == 0) return FRAMESIZE_QQVGA;
    if (strcasecmp(val, "hqvga") == 0 || strcasecmp(val, "240x176") == 0) return FRAMESIZE_HQVGA;
    if (strcasecmp(val, "qvga") == 0 || strcasecmp(val, "320x240") == 0) return FRAMESIZE_QVGA;
    if (strcasecmp(val, "cif") == 0 || strcasecmp(val, "400x296") == 0) return FRAMESIZE_CIF;
    if (strcasecmp(val, "hvga") == 0 || strcasecmp(val, "480x320") == 0) return FRAMESIZE_HVGA;
    if (strcasecmp(val, "vga") == 0 || strcasecmp(val, "640x480") == 0) return FRAMESIZE_VGA;
    char* endPtr = nullptr;
    long numeric = strtol(val, &endPtr, 10);
    if (endPtr != val && numeric >= 0 && numeric < FRAMESIZE_INVALID) {
      return static_cast<int>(numeric);
    }
    return -1;
  }

  // Persistent Multipart MJPEG stream: GET /stream
  static esp_err_t handleStream(httpd_req_t* req) {
    enableTcpNoDelay(req);
    applyQuerySensorParams(req);

    esp_err_t res = httpd_resp_set_type(req, "multipart/x-mixed-replace;boundary=frame");
    if (res != ESP_OK) return res;
    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
    httpd_resp_set_hdr(req, "Cache-Control", "no-store, no-cache, must-revalidate, max-age=0");
    httpd_resp_set_hdr(req, "X-Framerate", "20");

    char partHeader[128];
    while (true) {
      camera_fb_t* fb = esp_camera_fb_get();
      if (!fb) {
        vTaskDelay(pdMS_TO_TICKS(5));
        continue;
      }
      const uint32_t tsMs = static_cast<uint32_t>(esp_timer_get_time() / 1000ULL);
      const int hdrLen = snprintf(
          partHeader, sizeof(partHeader),
          "--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %u\r\nX-Timestamp-Ms: %u\r\n\r\n",
          static_cast<unsigned int>(fb->len), static_cast<unsigned int>(tsMs));

      res = httpd_resp_send_chunk(req, partHeader, hdrLen);
      if (res == ESP_OK) {
        res = httpd_resp_send_chunk(req, reinterpret_cast<const char*>(fb->buf), fb->len);
      }
      if (res == ESP_OK) {
        res = httpd_resp_send_chunk(req, "\r\n", 2);
      }
      esp_camera_fb_return(fb);
      if (res != ESP_OK) {
        break;
      }
    }
    return res;
  }

  // Persistent RFC 6455 Binary WebSocket stream: GET /ws
  static esp_err_t handleWebSocket(httpd_req_t* req) {
    enableTcpNoDelay(req);
    applyQuerySensorParams(req);

    char wsKey[64] = {0};
    if (httpd_req_get_hdr_value_str(req, "Sec-WebSocket-Key", wsKey, sizeof(wsKey)) != ESP_OK) {
      httpd_resp_send_err(req, HTTPD_400_BAD_REQUEST, "Missing Sec-WebSocket-Key");
      return ESP_FAIL;
    }

    String combined = String(wsKey) + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11";
    unsigned char sha1Out[20];
    mbedtls_sha1(reinterpret_cast<const unsigned char*>(combined.c_str()), combined.length(), sha1Out);
    unsigned char acceptB64[40] = {0};
    size_t outLen = 0;
    mbedtls_base64_encode(acceptB64, sizeof(acceptB64), &outLen, sha1Out, sizeof(sha1Out));

    const int fd = httpd_req_to_sockfd(req);
    if (fd < 0) return ESP_FAIL;

    char handshake[256];
    const int hsLen = snprintf(
        handshake, sizeof(handshake),
        "HTTP/1.1 101 Switching Protocols\r\n"
        "Upgrade: websocket\r\n"
        "Connection: Upgrade\r\n"
        "Sec-WebSocket-Accept: %s\r\n\r\n",
        reinterpret_cast<const char*>(acceptB64));
    if (send(fd, handshake, hsLen, 0) < 0) {
      return ESP_FAIL;
    }

    uint8_t wsHeader[10];
    while (true) {
      camera_fb_t* fb = esp_camera_fb_get();
      if (!fb) {
        vTaskDelay(pdMS_TO_TICKS(5));
        continue;
      }
      const size_t payloadLen = fb->len;
      size_t hdrBytes = 0;
      wsHeader[0] = 0x82;  // FIN + Binary frame opcode
      if (payloadLen <= 125) {
        wsHeader[1] = static_cast<uint8_t>(payloadLen);
        hdrBytes = 2;
      } else if (payloadLen <= 65535) {
        wsHeader[1] = 126;
        wsHeader[2] = static_cast<uint8_t>((payloadLen >> 8) & 0xFF);
        wsHeader[3] = static_cast<uint8_t>(payloadLen & 0xFF);
        hdrBytes = 4;
      } else {
        wsHeader[1] = 127;
        for (int i = 0; i < 8; ++i) {
          wsHeader[2 + i] = static_cast<uint8_t>((static_cast<uint64_t>(payloadLen) >> ((7 - i) * 8)) & 0xFF);
        }
        hdrBytes = 10;
      }

      bool ok = (send(fd, wsHeader, hdrBytes, 0) == static_cast<ssize_t>(hdrBytes));
      if (ok) {
        size_t sent = 0;
        while (sent < payloadLen) {
          ssize_t n = send(fd, fb->buf + sent, payloadLen - sent, 0);
          if (n <= 0) {
            ok = false;
            break;
          }
          sent += static_cast<size_t>(n);
        }
      }
      esp_camera_fb_return(fb);
      if (!ok) break;
    }
    return ESP_OK;
  }

  // Keep-Alive single frame capture: GET /capture
  static esp_err_t handleCapture(httpd_req_t* req) {
    enableTcpNoDelay(req);
    applyQuerySensorParams(req);

    camera_fb_t* fb = esp_camera_fb_get();
    if (!fb) {
      httpd_resp_send_500(req);
      return ESP_FAIL;
    }
    httpd_resp_set_type(req, "image/jpeg");
    httpd_resp_set_hdr(req, "Content-Disposition", "inline; filename=capture.jpg");
    httpd_resp_set_hdr(req, "Cache-Control", "no-store, no-cache");
    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
    esp_err_t res = httpd_resp_send(req, reinterpret_cast<const char*>(fb->buf), fb->len);
    esp_camera_fb_return(fb);
    return res;
  }

  // Compatible with Espressif CameraWebServer: GET /control?var=framesize&val=5
  static esp_err_t handleControl(httpd_req_t* req) {
    enableTcpNoDelay(req);
    char query[96];
    if (httpd_req_get_url_query_str(req, query, sizeof(query)) != ESP_OK) {
      httpd_resp_send_err(req, HTTPD_400_BAD_REQUEST, "Missing query");
      return ESP_FAIL;
    }
    char variable[32] = {0};
    char value[32] = {0};
    if (httpd_query_key_value(query, "var", variable, sizeof(variable)) != ESP_OK ||
        httpd_query_key_value(query, "val", value, sizeof(value)) != ESP_OK) {
      httpd_resp_send_err(req, HTTPD_400_BAD_REQUEST, "Require var and val");
      return ESP_FAIL;
    }
    sensor_t* s = esp_camera_sensor_get();
    if (!s) {
      httpd_resp_send_500(req);
      return ESP_FAIL;
    }
    int res = 0;
    if (strcmp(variable, "framesize") == 0) {
      int fs = parseFrameSize(value);
      if (fs < 0) {
        httpd_resp_send_err(req, HTTPD_400_BAD_REQUEST, "Invalid framesize");
        return ESP_FAIL;
      }
      if (s->pixformat == PIXFORMAT_JPEG) {
        res = s->set_framesize(s, static_cast<framesize_t>(fs));
      }
    } else if (strcmp(variable, "quality") == 0) {
      res = s->set_quality(s, atoi(value));
    } else if (strcmp(variable, "vflip") == 0) {
      res = s->set_vflip(s, atoi(value));
    } else if (strcmp(variable, "hmirror") == 0) {
      res = s->set_hmirror(s, atoi(value));
    }
    if (res != 0) {
      httpd_resp_send_500(req);
      return ESP_FAIL;
    }
    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
    return httpd_resp_send(req, "{\"ok\":true}", HTTPD_RESP_USE_STRLEN);
  }

  static esp_err_t handleStatus(httpd_req_t* req) {
    enableTcpNoDelay(req);
    auto* self = static_cast<FastStreamServer*>(req->user_ctx);
    sensor_t* s = esp_camera_sensor_get();
    if (!s) {
      httpd_resp_send_500(req);
      return ESP_FAIL;
    }
    const float ultraCm = self ? self->lastUltrasonicCm_ : -1.0f;
    const int m0 = self ? self->appliedStrengths_[0] : 0;
    const int m1 = self ? self->appliedStrengths_[1] : 0;
    const int m2 = self ? self->appliedStrengths_[2] : 0;
    const bool reflex = self ? self->reflexActive_ : false;
    char json[320];
    snprintf(json, sizeof(json),
             "{\"framesize\":%d,\"quality\":%d,\"vflip\":%d,\"hmirror\":%d,"
             "\"stream_path\":\"/stream\",\"ws_path\":\"/ws\",\"haptics_path\":\"/haptics\","
             "\"ultrasonic_cm\":%.1f,\"reflex_active\":%s,\"motor_strength_3\":[%d,%d,%d]}",
             s->status.framesize, s->status.quality, s->status.vflip, s->status.hmirror,
             ultraCm, reflex ? "true" : "false", m0, m1, m2);
    httpd_resp_set_type(req, "application/json");
    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
    return httpd_resp_send(req, json, HTTPD_RESP_USE_STRLEN);
  }

  // Bi-directional Haptics + Ultrasonic endpoint:
  // GET /haptics?l=80&c=0&r=0&ttl=1200  (or GET /haptics to read status/ultrasonic)
  static esp_err_t handleHaptics(httpd_req_t* req) {
    enableTcpNoDelay(req);
    auto* self = static_cast<FastStreamServer*>(req->user_ctx);
    if (!self) {
      httpd_resp_send_500(req);
      return ESP_FAIL;
    }
    char query[128];
    if (httpd_req_get_url_query_str(req, query, sizeof(query)) == ESP_OK) {
      char valBuf[16];
      int next[3] = {self->remoteStrengths_[0], self->remoteStrengths_[1], self->remoteStrengths_[2]};
      bool updated = false;
      const char* keys[3] = {"l", "c", "r"};
      for (int i = 0; i < 3; ++i) {
        if (httpd_query_key_value(query, keys[i], valBuf, sizeof(valBuf)) == ESP_OK) {
          int v = constrain(atoi(valBuf), 0, 100);
          next[i] = v;
          updated = true;
        }
      }
      uint32_t ttlMs = 1200;
      if (httpd_query_key_value(query, "ttl", valBuf, sizeof(valBuf)) == ESP_OK) {
        int parsedTtl = atoi(valBuf);
        if (parsedTtl >= 0 && parsedTtl <= 2500) {
          ttlMs = static_cast<uint32_t>(parsedTtl);
        }
      }
      if (updated) {
        const bool changed = (next[0] != self->remoteStrengths_[0] ||
                              next[1] != self->remoteStrengths_[1] ||
                              next[2] != self->remoteStrengths_[2]);
        for (int i = 0; i < 3; ++i) self->remoteStrengths_[i] = next[i];
        self->hapticsDeadlineMs_ = (ttlMs > 0 && (next[0] > 0 || next[1] > 0 || next[2] > 0))
                                       ? (millis() + ttlMs)
                                       : 0;
        self->applyMotorsNow();
        if (changed) {
          Serial.printf("Haptics [L,C,R]: [%d%%,%d%%,%d%%] PWM=[%d,%d,%d] ttl=%ums ultra=%.1fcm\n",
                        next[0], next[1], next[2],
                        getPWM(next[0]), getPWM(next[1]), getPWM(next[2]),
                        ttlMs, self->lastUltrasonicCm_);
        }
      }
    }
    const uint32_t now = millis();
    const int32_t remTtl = self->hapticsDeadlineMs_ > 0
                               ? static_cast<int32_t>(self->hapticsDeadlineMs_ - now)
                               : 0;
    char json[256];
    snprintf(json, sizeof(json),
             "{\"ok\":true,\"motor_strength_3\":[%d,%d,%d],"
             "\"ultrasonic_cm\":%.1f,\"reflex_active\":%s,\"ttl_ms\":%d}",
             self->appliedStrengths_[0], self->appliedStrengths_[1], self->appliedStrengths_[2],
             self->lastUltrasonicCm_, self->reflexActive_ ? "true" : "false",
             remTtl > 0 ? static_cast<int>(remTtl) : 0);
    httpd_resp_set_type(req, "application/json");
    httpd_resp_set_hdr(req, "Cache-Control", "no-store");
    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
    return httpd_resp_send(req, json, HTTPD_RESP_USE_STRLEN);
  }

  void ensureSensorTask() {
    if (sensorTaskHandle_ != nullptr) return;
    xTaskCreatePinnedToCore(sensorTaskEntry, "FastSensorHaptics", 3072, this, 1, &sensorTaskHandle_, 0);
  }

  static void sensorTaskEntry(void* arg) {
    auto* self = static_cast<FastStreamServer*>(arg);
    uint32_t lastUltraMs = 0;
    for (;;) {
      const uint32_t now = millis();
      if (self->ultraTrigPin_ >= 0 && self->ultraEchoPin_ >= 0 && (now - lastUltraMs >= 60)) {
        lastUltraMs = now;
        digitalWrite(self->ultraTrigPin_, LOW);
        delayMicroseconds(2);
        digitalWrite(self->ultraTrigPin_, HIGH);
        delayMicroseconds(10);
        digitalWrite(self->ultraTrigPin_, LOW);
        const long durationUs = pulseIn(self->ultraEchoPin_, HIGH, 15000);
        if (durationUs > 0) {
          self->lastUltrasonicCm_ = (durationUs * 0.0343f) / 2.0f;
        } else {
          self->lastUltrasonicCm_ = -1.0f;
        }
      }
      self->applyMotorsNow();
      vTaskDelay(pdMS_TO_TICKS(20));
    }
  }

  void applyMotorsNow() {
    const uint32_t now = millis();
    const bool remoteFresh = (hapticsDeadlineMs_ > 0 && static_cast<int32_t>(hapticsDeadlineMs_ - now) > 0);
    if (!remoteFresh) {
      remoteStrengths_[0] = 0;
      remoteStrengths_[1] = 0;
      remoteStrengths_[2] = 0;
    }
    const float d = lastUltrasonicCm_;
    reflexActive_ = (d > 2.0f && d <= ultraReflexCm_);
    int target[3] = {
      remoteStrengths_[0],
      reflexActive_ ? max(35, remoteStrengths_[1]) : remoteStrengths_[1],
      remoteStrengths_[2]
    };
    for (int i = 0; i < 3; ++i) {
      appliedStrengths_[i] = target[i];
      if (motorPins_[i] >= 0) {
        if (motorPwmAttached_[i]) {
          ledcWrite(motorPins_[i], getPWM(target[i]));
        } else {
          digitalWrite(motorPins_[i], target[i] > 0 ? HIGH : LOW);
        }
      }
    }
  }

  uint16_t port_;
  uint16_t streamPort_;
  httpd_handle_t cmdServer_ = nullptr;
  httpd_handle_t streamServer_ = nullptr;
  TaskHandle_t sensorTaskHandle_ = nullptr;
  int motorPins_[3] = {-1, -1, -1};
  bool motorPwmAttached_[3] = {false, false, false};
  uint32_t pwmFreq_ = 1000;
  uint8_t pwmResolution_ = 8;
  int remoteStrengths_[3] = {0, 0, 0};
  int appliedStrengths_[3] = {0, 0, 0};
  uint32_t hapticsDeadlineMs_ = 0;
  int ultraTrigPin_ = -1;
  int ultraEchoPin_ = -1;
  float ultraReflexCm_ = 60.0f;
  float lastUltrasonicCm_ = -1.0f;
  bool reflexActive_ = false;
};
