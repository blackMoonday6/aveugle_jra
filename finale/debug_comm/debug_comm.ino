

void setup()
{
 Serial.begin(9600);
 pinMode(trig, OUTPUT);
digitalWrite(trig, LOW);
pinMode(ad,OUTPUT);
pinMode(ag,OUTPUT);
pinMode(bd,OUTPUT);
pinMode(bg,OUTPUT);
pinMode(echo, INPUT);
pinMode(LED_BUILTIN, OUTPUT);

   
}
