#include <AccelStepper.h>

#define STEP_MULT 2

int inByte = 0;
char message[3];
AccelStepper stepper(AccelStepper::DRIVER, 8, 9);

void setup() {
  // put your setup code here, to run once
  stepper.setMaxSpeed(20000);
  stepper.setAcceleration(20000);
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
        switch (message[0]) {
          case 'f': // Forward
            stepper.move(STEP_MULT * long(message[1]));
            break;
          case 'b': // backward
            stepper.move(-STEP_MULT * long(message[1]));
            break;
          case 's': // set speed
            stepper.setMaxSpeed(float(message[1]));
            break;
        }

      }
    }    
  }
  stepper.run();
}
