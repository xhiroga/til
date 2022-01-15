import java.util.*;
import java.util.concurrent.*;

public class Fib5 {

	private ExecutorService exec = Executors.newCachedThreadPool();

	public class Task implements Callable<Integer> {

		private final int n;

		public Task(int n){
			this.n = n;
		}

		@Override
		public Integer call() throws Exception {
			if(n <= 1){
				return n;
			}
			Future<Integer> f1 = exec.submit(new Task(n-1));
			Future<Integer> f2 = exec.submit(new Task(n-2));
			return f1.get() + f2.get();
		}
}

	public void doExecute(){
		try{
			Future<Integer> f = exec.submit(new Task(10));
			System.out.println(f.get());
		}catch (Exception e){
			System.out.println(e);
		}finally{
			exec.shutdown();
		}
	}

	public static void main(String args[]){
		Fib5 f = new Fib5();
		f.doExecute();
	}

}
