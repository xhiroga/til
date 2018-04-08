// 温度に応じてLEDライトを光らせる

const int sensorPin = A0;
const float baselineTemp = 12.0;

void setup() {
   Serial.begin(9600); // シリアルポートを開く。9600bits/second
   for(int pinNumber = 2; pinNumber<5; pinNumber++){
    pinMode(pinNumber, OUTPUT);
    digitalWrite(pinNumber, LOW);  
  }
}

void loop() {
  int sensorVal = analogRead(sensorPin);
  Serial.print("Sensor Value: "); // シリアルポートで接続されているコンピューターに出力
  Serial.print(sensorVal);

  float voltage = (sensorVal/1024.0) * 5.0;
  Serial.print(", Volts: ");
  Serial.print(voltage);

  float temperture = (voltage - .5) * 100;
  Serial.print(", degrees C: ");
  Serial.println(temperture);

  if(temperture < baselineTemp) {
    digitalWrite(2, LOW);
    digitalWrite(3, LOW);
    digitalWrite(4, LOW);
  } else if (temperture > baselineTemp &&
    temperture < baselineTemp + 4){
    digitalWrite(2, HIGH);
    digitalWrite(3, LOW);
    digitalWrite(4, LOW);
  } else if (temperture > baselineTemp + 4 &&
    temperture < baselineTemp + 8){
    digitalWrite(2, HIGH);
    digitalWrite(3, HIGH);
    digitalWrite(4, LOW);
  }else if (temperture > baselineTemp + 8 &&
    temperture < baselineTemp + 12){
    digitalWrite(2, HIGH);
    digitalWrite(3, HIGH);
    digitalWrite(4, HIGH);
  }
  delay(1);  
}
