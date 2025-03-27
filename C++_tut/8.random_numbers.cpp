#include <iostream>

int main(){

    //pseudo-random 
    
    srand(time(NULL)); //time(NULL) used as seed

    int max_num = 20; //to generate random numbers up to max_num 
    
    int num = rand()%max_num; //from  0 to max_num -1
    int num1 = (rand()%max_num) + 1; //from  1 to max_num

    std::cout<<num;

    return 0;
}