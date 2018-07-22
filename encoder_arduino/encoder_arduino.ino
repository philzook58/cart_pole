// Red - 5V
// Black - GND
const int encoder_1i = 2; // Green interrupt
const int encoder_1b = 4; // White
const int encoder_2i = 3; // Green interrupt
const int encoder_2b = 5; // White interrupt
const int limit_1 = 8; // 
const int limit_2 = 9; // 
long encoder_1 = 0;
long encoder_1_change = 0;
long encoder_2 = 0;

void setup() {
Serial.begin(115200, SERIAL_8E1);
pinMode(encoder_1i, INPUT_PULLUP);
pinMode(encoder_2i, INPUT_PULLUP);
pinMode(encoder_1b, INPUT_PULLUP);
pinMode(encoder_2b, INPUT_PULLUP);
pinMode(limit_1, INPUT_PULLUP);
pinMode(limit_2, INPUT_PULLUP);

attachInterrupt(0, encoder1PinChange, CHANGE);
attachInterrupt(1, encoder2PinChange, CHANGE);
}

void loop() {
//Serial.write(String(encoder_1) + "\t" + String(encoder_2) + "\t" + String(limit_1) + "\t" + String(limit_2))
  if (Serial.available()) {
    Serial.read();
    Serial.print(encoder_1);
    Serial.print("\t");
    Serial.print(encoder_2);
    Serial.print("\t");
    Serial.print(!digitalRead(limit_1));
    Serial.print("\t");
    Serial.println(!digitalRead(limit_2));
    Serial.flush();
  }
}

void encoder1PinChange() {
encoder_1 += digitalRead(encoder_1i) == digitalRead(encoder_1b) ? -1 : 1;
encoder_1_change++;
}

void encoder2PinChange() {
encoder_2 += digitalRead(encoder_2i) != digitalRead(encoder_2b) ? -1 : 1;
}
