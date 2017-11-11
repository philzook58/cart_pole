#include <AccelStepper.h>

#define MULT 1000.

int inByte = 0;
char message[3];
AccelStepper stepper(AccelStepper::DRIVER, 8, 9);

void setup() {
  // put your setup code here, to run once
  stepper.setMaxSpeed(200000);
  stepper.setAcceleration(200000);
  Serial.begin(115200);
  while (!Serial) {
    ; // wait for serial port to connect. Needed for native USB port only
  }
}



void loop() {
  // put your main code here, to run repeatedly:
  if (Serial.available() > 0) {
    inByte = Serial.read();
    if (inByte == '^') { // starting character of message is ^
      Serial.readBytes(message, 3);
      if (message[2] == ';' ) {
        float accel =  MULT * float(message[1]);
        switch (message[0]) {
          case 'f': // Forward
           if (accel != 0.) {
              stepper.setAcceleration(accel);
              Serial.println(accel);
              stepper.move(100);
            } else {
              stepper.move(0);
            }
            break;
          case 'b': // backward
           if (accel != 0.) {
              stepper.setAcceleration(accel);
              Serial.println(-1*accel);
              stepper.move(-100);
            } else {
              stepper.move(0);
            }
            break;
        }

      }
    }    
  }
  stepper.run();
}

