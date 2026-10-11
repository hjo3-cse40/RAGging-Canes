#pragma once
// Optional active-high PWM driver adapter for ESP32 Arduino core 3.x.
// Configure ONLY after checking motor driver ratings and camera LEDC resources.
#include "HapticClient.h"
#include <driver/gpio.h>
#include <soc/soc_caps.h>

struct HapticMotorConfig {
  int pin = -1;              // -1 disables this motor; no default board pins.
  uint8_t channel = 255;     // Explicit LEDC channel, not allocated automatically.
  uint8_t minDuty = 0;       // Calibrate lowest useful duty (8-bit: 0..255).
  uint8_t maxDuty = 0;       // Zero disables electrical output until calibrated.
};

class HapticMotors {
 public:
  HapticMotors(HapticClient& client, const HapticMotorConfig (&config)[3], uint32_t pwmHz)
      : client_(client), pwmHz_(pwmHz) {
    for (int i = 0; i < 3; ++i) config_[i] = config[i];
  }

  bool begin() {
    if (task_) return true;
    if (!pwmHz_) return false;
    for (int i = 0; i < 3; ++i) {
      const auto& c = config_[i];
      if (c.pin < 0) continue;
      if (!GPIO_IS_VALID_OUTPUT_GPIO(c.pin) || c.channel >= SOC_LEDC_CHANNEL_NUM || c.minDuty > c.maxDuty) return false;
      for (int j = 0; j < i; ++j) {
        if (config_[j].pin >= 0 && (config_[j].pin == c.pin || config_[j].channel == c.channel)) return false;
      }
    }
    for (int i = 0; i < 3; ++i) {
      const auto& c = config_[i];
      if (c.pin < 0) continue;
      pinMode(c.pin, OUTPUT);
      digitalWrite(c.pin, LOW);
      if (!ledcAttachChannel(c.pin, pwmHz_, 8, c.channel)) { detach(); return false; }
      attached_[i] = true;
      if (!ledcWrite(c.pin, 0)) { detach(); return false; }
    }
    if (xTaskCreate(taskEntry, "haptic-motors", 3072, this, 1, &task_) != pdPASS) {
      detach();
      return false;
    }
    return true;
  }

  static uint8_t dutyFor(int strength, const HapticMotorConfig& c) {
    if (strength <= 0 || c.maxDuty == 0) return 0;
    if (strength > 100) strength = 100;
    return c.minDuty + (static_cast<unsigned>(strength) * (c.maxDuty - c.minDuty) + 50) / 100;
  }

 private:
  void detach() {
    for (int i = 0; i < 3; ++i) if (attached_[i]) {
      ledcWrite(config_[i].pin, 0);
      ledcDetach(config_[i].pin);
      digitalWrite(config_[i].pin, LOW);
      attached_[i] = false;
    }
  }

  static void taskEntry(void* pointer) {
    auto* self = static_cast<HapticMotors*>(pointer);
    for (;;) {
      int strengths[3];
      self->client_.read(strengths); // Expired/disconnected commands produce zero immediately.
      for (int i = 0; i < 3; ++i) if (self->attached_[i]) {
        if (!ledcWrite(self->config_[i].pin, dutyFor(strengths[i], self->config_[i]))) {
          self->detach(); // Fail closed until firmware is restarted.
          vTaskDelete(nullptr);
          return;
        }
      }
      vTaskDelay(pdMS_TO_TICKS(10));
    }
  }

  HapticClient& client_;
  HapticMotorConfig config_[3];
  uint32_t pwmHz_;
  bool attached_[3] = {false, false, false};
  TaskHandle_t task_ = nullptr;
};
