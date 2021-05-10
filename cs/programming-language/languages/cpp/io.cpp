#include<iostream>
#include<fstream>
#include<string>

using namespace std;
int main(){
	string line;

	ofstream myfileI("io.txt", ios::app);
	if(myfileI.is_open())
	{
		myfileI << "I am adding the line\n";
		myfileI << "I am adding the another line\n";
		myfileI.close();
	}
	else cout << "Unable to open file for writing";

	ifstream myfileO("io.txt");
	if(myfileO.is_open())
	{
		while(getline(myfileO, line)) // lineに一行渡すとともにbooleanを返す.
		{
			cout << line << "\n";
		}
		myfileO.close();
	}
	else cout << "unable to open for reading";

	return 0;
}
