// ポテンショメーターの入力に応じてライトを光らせる！

const int potPin = A0;
int potVal = 0;
const int redLight = 13;
const int yellowLight = 12;
const int greenLight = 11;
const int blueLight = 10;

void setup() {
  Serial.begin(9600);
  pinMode(redLight, OUTPUT);
  pinMode(yellowLight, OUTPUT);
  pinMode(greenLight, OUTPUT);
  pinMode(blueLight, OUTPUT);
}

void loop() {
  resetLight();
  
  potVal = analogRead(potPin);
  Serial.print("Potentiometer Value: ");
  Serial.println(potVal);

  if (potVal >= 255){
    digitalWrite(blueLight, HIGH);
    Serial.println("BLUE!");
  }
  if (potVal >= 511){
    digitalWrite(greenLight, HIGH);
    Serial.println("GREEN!");
  }
  if (potVal >= 767){
    digitalWrite(yellowLight, HIGH);
    Serial.println("YELLOW!");
  }
  if (potVal >= 1023){ // 1023ぴったりにはならないと思われるため
    digitalWrite(redLight, HIGH);
    Serial.println("RED!");
  }
  delay(150);
}

void resetLight(){
  digitalWrite(redLight, LOW);
  digitalWrite(yellowLight, LOW);
  digitalWrite(greenLight, LOW);
  digitalWrite(blueLight, LOW);
}