
object Question7a{
	class BankAccount {
		private var _balance = 0.00
		def this(n: Double){
			this()
			if(n >= 0){
				_balance = n
			}
			else{
				println("This is not tangible money. \n Current balance is reset to 0.00.")
			}
		}

		def currentBalance = _balance
		def deposit(d: Double){
			_balance = _balance + d
		}
		def withdraw(w: Double){
			if(w <= _balance){
				_balance = _balance - w
			}
			else{
				println("You don't have this amount of money.")
			}
		}
	}
	def main(args: Array[String]){
		println("Instantiate Darshan's account with $100.")
		var darshan = new BankAccount(100)
		println("Current Balance: $" + darshan.currentBalance)
		println("Add $5.")
		darshan.deposit(5)
		println("Current Balance: $" + darshan.currentBalance)
		println("Withdraw $1000.")
		darshan.withdraw(1000)
		println("Current Balance: $" + darshan.currentBalance)
		println("Withdraw $12.95")
		darshan.withdraw(12.95)
		println("Current Balance: $" + darshan.currentBalance)
	}
}
