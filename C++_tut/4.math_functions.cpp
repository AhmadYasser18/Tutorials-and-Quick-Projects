#include <iostream>
#include <cmath>

int main(){

    int x = 10;
    int y = 25;

    std::cout<<"Max: "<<std::max(x, y)<<std::endl;
    std::cout<<"Min: "<<std::min(x, y)<<std::endl;
    
    std::cout<<"2^3: "<<pow(2,3)<<std::endl;
    std::cout<<"root 25: "<<sqrt(25)<<std::endl;
    std::cout<<"Absolute(-5): "<<abs(-5)<<std::endl;
    std::cout<<"Rounding 2.55: "<<round(2.55)<<std::endl;
    std::cout<<"Rounding up 2.15: "<<ceil(2.15)<<std::endl;
    std::cout<<"Rounding down 2.55: "<<floor(2.55)<<std::endl;


    return 0;
}