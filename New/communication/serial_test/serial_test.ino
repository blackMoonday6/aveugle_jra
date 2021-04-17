int angle;
int person;
boolean pers;
String ser;
String sub;
int index;

void setup()
{

  Serial.begin(9600);
  pinMode(LED_BUILTIN,OUTPUT);
}





void loop()
{
if(Serial.available()>0)
{
  ser=Serial.readStringUntil('\n');
  index = ser.indexOf(','); 
  sub=ser.substring(0,index);
  angle=sub.toInt();
  sub=ser.substring(index+1);
  person=sub.toInt();
  if(person==1)
  {
    pers=true;
    Serial.write("there is a person");
      Serial.write("yes");
        digitalWrite(LED_BUILTIN,HIGH);
  
  }
    if(person==0)
  {
    pers=false;
    Serial.write("there is not a pereson");
      Serial.write("no");
      digitalWrite(LED_BUILTIN,LOW);
   
  
  }
  
}
  
  
}
