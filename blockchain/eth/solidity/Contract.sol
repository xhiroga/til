pragma solidity ^0.4.19;
// 初めにsolidityのバージョン宣言を行う


contract ZombieFactory {

    event NewZombie(uint zombieId, string name, uint dna);
    // contractのフィールドに定義し、functionで発火させ、web3.jsでlistenする。

    uint dnaDigits = 16;
    uint dnaModulus = 10 ** dnaDigits;
    // uint... unsigned integerのこと（符号なし整数）

    struct Zombie {
        string name;
        uint dna;
    }

    Zombie[] public zombies;

    mapping (uint => address) public zombieToOwner;
    mapping (address => uint) ownerZombieCount;

    function createRandomZombie(string _name) public {
        require(ownerZombieCount[msg.sender] == 0); // if文とthorwの合わせ技的な文法。
        uint randDna = _generateRandomDna(_name);
        _createZombie(_name, randDna);
    }

    // private関数は_で始めるのが通例
    function _createZombie(string _name, uint _dna) private {
        uint id = zombies.push(Zombie(_name, _dna)) - 1;
        zombieToOwner[id] = msg.sender;
        ownerZombieCount[msg.sender]++;
        NewZombie(id, _name, _dna);
    }

    // return ではなく returns
    // アプリケーション内のデータを更新できない関数はviewを、参照さえできない関数はpureを宣言する。
    function _generateRandomDna(string _str) private view returns (uint) {
        uint rand = uint(keccak256(_str));
        return rand % dnaModulus;
    }

}
