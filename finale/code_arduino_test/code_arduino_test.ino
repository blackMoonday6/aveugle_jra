int trig = 2;
int echo = 3;
int ad=4;//moteur avant droite
int ag=6;//mouteur avant gauche
long lecture_echo;
long cm;
int  vitesse;
int angle;
int velocity;
int positionn;
String reciveur;
String reciveur2;
int index;
void setup(){
 
pinMode(trig, OUTPUT);
digitalWrite(trig, LOW);
pinMode(ad,OUTPUT);
pinMode(ad,OUTPUT);
pinMode(echo, INPUT);
Serial.begin(9600);


}

void loop(){


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
  
  
  
  
  
digitalWrite(trig, HIGH);
delayMicroseconds(10);
digitalWrite(trig, LOW);
lecture_echo = pulseIn(echo,HIGH);
cm = lecture_echo /58;

delay(15);
if(cm<50){
vitesse=map(cm,1,50,255,60);


}

if((positionn>282&&positionn<312)&&(cm<50)){
  
analogWrite(ag,vitesse);
analogWrite(ad,vitesse);

}

if((cm>50)){
  vitesse=0;
  analogWrite(ag,vitesse);
analogWrite(ad,vitesse);
}

}
}
