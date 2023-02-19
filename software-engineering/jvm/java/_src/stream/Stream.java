import java.util.*;

public class Stream{
	public static void main(String[] args){
			List<String> list = Arrays.asList("a1","a2","a3","b4","c5","c6");

			list
				.stream()
				.filter(s -> s.startsWith("c"))
				.map(String::toUpperCase)
				.forEach(System.out::println);
		}

	}
