import java.util.List;
import java.util.ArrayList;

public class Check{
	public static void main(String args[]){
		List<Person> persons = new ArrayList<Person>();

	Person p1 = new Person();
	p1.setAge(12);
	Person p2 = new Person();
	p2.setAge(22);

	persons.add(p1);
	persons.add(p2);

	boolean foo = persons.stream().anyMatch(p -> p.getAge() < 20);

	System.out.println("hey");
	System.out.println(foo);


	}
}
