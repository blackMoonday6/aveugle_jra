int angle;
int velocity;
int positionn;
boolean work;
String reciveur;
String reciveur2;
int trig = 2;
int echo = 3;
int ad=4;//moteur avant droite
int ag=7;//mouteur avant gauche
int bd=5;//moteur avant droite
int bg=6;
long lecture_echo;
long cm;

int  vitesse;
int maxvitesse;
int index;
int see;
void setup()
{
 Serial.begin(9600);
 pinMode(trig, OUTPUT);
digitalWrite(trig, LOW);
pinMode(ad,OUTPUT);
pinMode(ad,OUTPUT);
pinMode(echo, INPUT);
pinMode(LED_BUILTIN, OUTPUT);

   
}
void loop()
{
    digitalWrite(trig, HIGH);
    delayMicroseconds(10);
    digitalWrite(trig, LOW);
    lecture_echo = pulseIn(echo,HIGH);
    cm = lecture_echo /58;
    delay(50);
    if((cm>74)||(cm==0))
    {vitesse =0;}
    else{
    vitesse=map(cm,75,120,75,0);}
    
if(Serial.available()>0)
{
  reciveur=Serial.readStringUntil('\n');
  positionn=reciveur.toInt();}
/*reciveur2=reciveur.substring(0,index);
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
  positionn=reciveur2.toInt();*/
if(cm<75){
  if((positionn>270)&&(positionn<295)){  /*Milieu*/
    analogWrite(ad,vitesse);
    analogWrite(ag,vitesse);
    analogWrite(bd,0);
    analogWrite(bg,0);
    digitalWrite(LED_BUILTIN, HIGH);
    
  
  }
    
  if((positionn>220)&&(positionn<271)){ /*Droite*/
    analogWrite(ad,vitesse);
    analogWrite(ag,vitesse*0.7);//position/314
    analogWrite(bd,0);
    analogWrite(bg,0);
    digitalWrite(LED_BUILTIN,LOW);}
    
  if((positionn>294)&&(positionn<370)){ /*Gauche*/
    analogWrite(ad,vitesse*0.6);
    analogWrite(ag,vitesse*1.2);
    analogWrite(bd,0);
    analogWrite(bg,0);
    digitalWrite(LED_BUILTIN,LOW);}
 /* 
  if ((positionn>370)||(positionn<294))
  {
  analogWrite(ad,0);
    analogWrite(bd,0);
    analogWrite(bg,0);
    analogWrite(ag,0);
    Serial.println(positionn);
  digitalWrite(LED_BUILTIN,LOW);}*/
  if ((positionn>369))/*barcha Gauche*/
  {
    analogWrite(ad,vitesse*0.4);
    analogWrite(ag,vitesse*1.3);
    analogWrite(bg,0);
    analogWrite(bd,0);
  digitalWrite(LED_BUILTIN,LOW);}
  if ((positionn<221))/*barcha Droite*/
  {
  analogWrite(ad,vitesse*1.2);
    analogWrite(ag,vitesse*0.6);
    analogWrite(bg,0);
    analogWrite(ag,0);
  digitalWrite(LED_BUILTIN,LOW);}
}
else{analogWrite(ad,0);
    analogWrite(bd,0);
    analogWrite(bg,0);
    analogWrite(ag,0);
  digitalWrite(LED_BUILTIN,LOW);
}


} 

 

  

