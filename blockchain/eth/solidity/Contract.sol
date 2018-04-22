pragma solidity ^0.4.19;
// 初めにsolidityのバージョン宣言を行う


contract ZombieFactory {

    uint dnaDigits = 16;
    uint dnaModulus = 10 ** dnaDigits;
    // uint... unsigned integerのこと（符号なし整数）

    struct Zombie {
        string name;
        uint dna;
    }

    Zombie[] public zombies;

    // private関数は_で始めるのが通例
    function _createZombie(string _name, uint _dna) private {
        zombies.push(Zombie(_name, _dna));
    }

    // return ではなく returns
    // アプリケーション内のデータを更新できない関数はviewを、参照さえできない関数はpureを宣言する。
    function _generateRandomDna() private view returns(uint){
        
    }

}
