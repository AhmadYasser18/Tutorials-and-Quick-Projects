#include <iostream>

int main(){

    srand(time(0));

    int tries;
    int guess;
    int num = (rand()%50) +1; //numbers from 1-50

    tries = 0;
    do{
        std::cout<<"Enter a number: ";
        std::cin>>guess;

        if (guess > num){
            std::cout<<"High! Go Lower.\n\n";
        }
        else if (guess < num){
            std::cout<<"Low! Go Higher.\n\n" ;           
        }
        else if (num == 0){
            std::cout<<"The number is: "<<num ;
            break;
        }
        else{
            std::cout<<"Correct the number is: "<<num<<"\n\nYou got it in "<<tries<<" tries.";
            
        }
        tries++; 

    }
    while(guess != num);



    return 0;
}