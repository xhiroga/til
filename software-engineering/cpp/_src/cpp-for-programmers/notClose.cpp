#include <iostream>
#include <fstream>

using namespace std;

int main()
{
	ofstream myfile ("new.txt", ios::app);
	cout << "open and not close. what will happen?";
	// 何か起きるかと思ったが、特に何も起きなかった。

	return 0;
}
