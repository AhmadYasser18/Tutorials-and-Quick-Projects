#include <iostream>

int main(){

    //while 
    std::string name;

    while (name.empty()){
        std::cout<<"Enter your name: ";
        std::getline(std::cin, name);
    }

    std::cout<<"Hello "<<name<<std::endl;
    ///////////////
    //do-while

    int num;

    do{
        std::cout<<"Enter a number: ";
        std::cin>>num;
    }
    while (num<0);

    std::cout<<num<<std::endl;

    ///////////////
    /*for :-to loop a certain number of times
           -up to 3 statements could be added in ():
                - counter
                - stopping condition
                - increment/decrement
    */
    for (int i = 1; i < 5; i++){
        std::cout<<i<<": "<<i*2<<std::endl;
    }


    return 0;
}