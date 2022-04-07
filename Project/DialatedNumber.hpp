// Kody Manastyrski
// CP 8316
// Computational Geometry
// Dialated Number hpp
//

#include<bitset>

using namespace std;

class Dialated{

	protected:
		int value;
		bitset<64> mask;

	public:
		Dialated(int val);
		Dialated operator+(const Dialated& other);
		Dialated operator-(const Dialated& other);
		Dialated operator-();
		bool operator==(const Dialated& other);
		int to_int();


};
