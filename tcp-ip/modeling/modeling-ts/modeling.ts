/**
 * data in physical layer
 * @see https://en.wikipedia.org/wiki/Ethernet_frame
 */

type EthernetPacket = {
  header: {
    preamble: ArrayBuffer;
    sfd: ArrayBuffer;
  };
  payload: Frame;
};

type MacAddress = ArrayBuffer;

/**
 * data in datalink layer
 * @see https://en.wikipedia.org/wiki/Ethernet_frame
 */
type Frame = {
  header: {
    macDestination: MacAddress;
    macSource: MacAddress;
    ieee802_1QTag: ArrayBuffer;
    etherTypeOrLength: ArrayBuffer;
  };
  payload: IpV4Packet;
  frameCheckSequence: ArrayBuffer;
};

type IpAddress = ArrayBuffer;

/**
 * data in network layer
 * @see https://ja.wikipedia.org/wiki/IPv4
 */
type IpV4Packet = {
  header: {
    version: 4;
    ihl: 5;
    typeOfService: {};
    totalLength: number;
    identification: ArrayBuffer;
    variousControlFlag: 0 | 1;
    fragmentOffset: ArrayBuffer;
    timeToLive: number;
    protocol: "ICMP" | "TCP" | "UDP" | "EIGRP" | "OSPF"; // IPv6もここに入るの？
    checkSum: ArrayBuffer;
    sourceAddress: IpAddress;
    destinationAddress: IpAddress;
    options: ArrayBuffer;
  };
  payload: Segment | UdpDatagram; // and more...
};

/**
 * data in tcp
 */
type Segment = {};

/**
 * data in udp
 */
type UdpDatagram = {};

type PDU = EthernetPacket | Frame | IpV4Packet | Segment;

/**
 * @see https://en.wikipedia.org/wiki/OSI_model
 */
interface Device {
  macAddress: MacAddress;
  receive: (packet: EthernetPacket) => void;
}

class Computer implements Device {
  macAddress: MacAddress;
  constructor(macAddress: MacAddress) {
    this.macAddress = macAddress;
  }
  receive = (packet: EthernetPacket) => {
    if (packet.payload.header.macDestination === this.macAddress) {
      console.log(packet);
    } else {
      // drop
    }
  };
}

/**
 * @see https://amzn.to/3db1h3W - 実践 パケット解析 第3版 ―Wiresharkを使ったトラブルシューティング
 */
class Hub {
  ports: Device[];
  constructor(ports: Device[]) {
    this.ports = ports;
  }
  receive = (packet: EthernetPacket) => {
    this.ports.forEach((device) => device.receive(packet));
  };
}

/**
 * references
 * @see https://packet6.com/cam-table-fundamental-switch-operations/
 */
type ContentAddressableMemoryTable = {
  port: Device;
  macAddress: MacAddress;
}[];
type CamTable = ContentAddressableMemoryTable;

/**
 * @see https://amzn.to/3db1h3W - 実践 パケット解析 第3版 ―Wiresharkを使ったトラブルシューティング
 */
class Switch {
  ports: Device[];
  camTable: CamTable;
  constructor(ports: Device[]) {
    this.ports = ports;
    this.camTable = ports.map((device) => ({
      port: device,
      macAddress: device.macAddress,
    }));
  }
  receive = (packet: EthernetPacket) => {
    const macDestination = packet.payload.header.macDestination;
    this.camTable
      .filter((entry) => entry.macAddress === macDestination)
      .forEach((entry) => entry.port.receive(packet));
  };
}

type RoutingTable = {};
class Router {
  wanPort: Device;
  lanPorts: Device[];
  routingTable: RoutingTable;
  constructor(wanPort: Device, lanPorts: Device[]) {}
  receive = () => {
    // TODO
  };
}
