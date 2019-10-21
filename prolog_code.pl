countDiseases(X,Y):- length(X,Y).
noCriticalDisease(X):- not(member(hearing,X)).
mentalHealth(X):- member(mental,X).

risk(Age,low,Diseases,Reduce):- reduce(Reduce), risk(Age,medium,Diseases,_),!.

risk(Age,low,Diseases,_) :- countDiseases(Diseases,Y), Y<2, 
    						(Age<55, noCriticalDisease(Diseases); 
    						( not(mentalHealth(Diseases)),!)).

risk(Age,medium,Diseases,_):-  noCriticalDisease(Diseases), (   
                               (age(Age,X), X = over55, countDiseases(Diseases,Y),Y<3,!); 
                               (Age<55, countDiseases(Diseases,2),!);
    						   (not(mentalHealth(Diseases)) , countDiseases(Diseases,2),!)).
    						 
risk(Age,rejected,Diseases,_):- risk(Age,high,Diseases,_),!.

risk(Age,high,Diseases,_):-(    member(hearing,Diseases);(countDiseases(Diseases,X), X>2));
    						(   mentalHealth(Diseases), age(Age,X), X = over65).

age(Age, over55):- Age>55, Age<66.
age(Age, acceptable):- Age>20,Age<71.
age(Age, over65):- Age>65, Age<71.

country(switzerland).
reduce(yes).

findRisk(Age,Risk,Diseases,Country,Reduce):- risk(Age,Risk, Diseases,Reduce), country(Country), age(Age, acceptable).

