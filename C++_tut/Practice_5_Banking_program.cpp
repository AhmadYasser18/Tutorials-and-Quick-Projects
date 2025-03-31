#include <iostream>
#include <iomanip>

float deposit(float balance);
float withdraw(float balance);
void show_balance(float balance);

int main(){

    std::string name;
    float balance = 0.00;

    float amount;
    
    int op;
    bool exit = 1;
    do
    {
        std::cout<<"Hello user!\nWhat would you like to do?\n1:Deposit  2:Withdraw  3:Check Balance\n";
        std::cin>>op;

        while (op!= 1 && op!=2 && op!=3)
        {
            std::cout<<"Please enter one of the following?\n1:Deposit  2:Withdraw  3:Check Balance\n";
            std::cin>>op;
        }

        switch (op)
        {
        case 1:
            balance = deposit(balance);
            show_balance(balance);
            break;
        
        case 2:
            balance = withdraw(balance);
            show_balance(balance);
            break;
        
        case 3:
            show_balance(balance);
            break;
        
        default:
            break;
        }

        std::cout<<"Do you want to do any more transactions?\nY:1   N:0 ";
        std::cin>>exit;

    } while (exit);

    std::cout<<"Thank You.";

    return 0;
}


float deposit(float balance){

    float amount;

    std::cout<<"Please enter amount to be deposited: ";
    std::cin>>amount;

    while( amount < 0){
        std::cout<<"Amount can't be in negative";
        std::cin>>amount;        
    }

    balance+=amount;
    std::cout<<amount<<" have been successfully deposited.\n";

    return balance;
}

float withdraw(float balance){

    if (balance == 0){
        std::cout<<"Empty Balance.\n";
        return 0;
    }
    
    float amount;

    std::cout<<"\nPlease enter amount to be withdrawn: ";
    std::cin>>amount;
    
    while( amount < 0 || amount > balance){
        if (amount<0)
        {
            std::cout<<"Amount can't be in negative";
            std::cin>>amount;
        }

        if (amount > balance)
        {
            std::cout<<"Insufficient balance. Enter another amount:";
            std::cin>>amount;
        }

    }
    balance-=amount;
    std::cout<<amount<<"have been successfully withdrawn.\n";

    return balance;
}

void show_balance(float balance){
    std::cout<<"\nYour remaining balance is: "<<std::setprecision(2)<< std::fixed<< balance<<std::endl<<std::endl;
}