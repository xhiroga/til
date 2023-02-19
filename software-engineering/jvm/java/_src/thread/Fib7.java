import java.util.concurrent.RecursiveTask;
import java.util.concurrent.ForkJoinPool;

public class Fib7 extends RecursiveTask<Integer>{

	private final int n;

	Fib7(int n){
		this.n = n;
	}

	@Override
	protected Integer compute(){
		if(n <= 1){
			return n;
		}
		Fib7 f1 = new Fib7(n - 1);
		f1.fork();

		Fib7 f2 = new Fib7(n - 2);
	return f2.compute() + f1.join();

	}

	public static void main(String args[]){
		ForkJoinPool pool = new ForkJoinPool();
		System.out.println(pool.invoke(new Fib7(45)));
	}

}
