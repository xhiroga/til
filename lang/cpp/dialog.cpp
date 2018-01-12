#include <iostream>
#include <string>

int main()
{
	int year = 0;
	int age = 0;
	std::string name = " ";

	std::cout << "what is your birth year?\n";
	std::cin >> year;

	std::cout << "what is your age?\n";
	std::cin >> age;

	std::cout << "what is your name?\n";
	std::cin >> name;

	std::cout << "You born in " << year << " and now "
	<< age <<" years old, your name is " << name << "!\n";
}
