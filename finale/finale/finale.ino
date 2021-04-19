int angle;
int velocity;
int positionn;
String reciveur;
String reciveur2;
int trig = 2;
int echo = 3;
int ad=4;//moteur avant droite
int ag=5;//mouteur avant gauche
long lecture_echo;
long cm;
int  vitesse;
int maxvitesse;
int index;
void setup()
{
 Serial.begin(9600);
 pinMode(trig, OUTPUT);
digitalWrite(trig, LOW);
pinMode(ad,OUTPUT);
pinMode(ad,OUTPUT);
pinMode(echo, INPUT);
   
}
void loop()
{
  

if(Serial.available()>0)
{
  reciveur=Serial.readStringUntil('\n');
  index = reciveur.indexOf(','); 

  reciveur2=reciveur.substring(0,index);
  angle=reciveur2.toInt();
  reciveur2=reciveur.substring(index+1);
 reciveur=reciveur2;
 index = reciveur.indexOf(',');
 reciveur2=reciveur.substring(0,index);
  velocity=reciveur2.toInt();
   reciveur2=reciveur.substring(index+1);
 reciveur=reciveur2;
 index = reciveur.indexOf(',');
 reciveur2=reciveur.substring(0,index);
  positionn=reciveur2.toInt();
if(velocity==2)
{
maxvitesse-=10;
}
if(velocity==3)
{
maxvitesse+=10;
}
 if(cm>100){
  vitesse=0;
}

analogWrite(ag,vitesse);
Serial.println(vitesse);
delay(100);
digitalWrite(trig, HIGH);
delayMicroseconds(10);
digitalWrite(trig, LOW);
lecture_echo = pulseIn(echo,HIGH);
cm = lecture_echo /58;

delay(15);
if(cm<100){
vitesse=map(cm,1,maxvitesse,255,60);


}
if(cm>100||velocity==0){
  vitesse=0;
}


velocity=1;






}


} 

 

  

