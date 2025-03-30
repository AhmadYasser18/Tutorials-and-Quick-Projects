#include <iostream>

float deposit(float balance);
float withdraw(float balance);
void show_balance(float balance);

int main(){

    std::string name;
    float balance = 0;

    float amount;
    
    int op;
    bool exit = 1;
    do
    {
        std::cout<<"Hello user!\nWhat would you like to do?/n1:Deposit  2:Withdraw  3:Check Balance\n";
        std::cin>>op;

        switch (op)
        {
        case 1:
            balance = deposit(balance);
            break;
        
        case 2:
            balance = withdraw(balance);
            break;
        
        case 3:
            show_balance(balance);
            break;
        
        default:
            break;
        }

        std::cout<<"Do you any more transactions?\nY:1, N:0 ";
        std::cin>>exit;

    } while (exit);

    return 0;
}


float deposit(float balance){

    std::cout<<"Your remaining balance is: "<<balance;
    return balance;
}

float withdraw(float balance){
    
    std::cout<<"Your remaining balance is: "<<balance;
    return balance;
}

void show_balance(float balance){
    std::cout<<"Your remaining balance is: "<<balance;
}