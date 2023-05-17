import urx
import math 
import time
from pyModbusTCP.client import ModbusClient
import CONFIG

class RobotMB:
    """Interface to exchange RTDE for Modbus
    reimplements all robot functions previously performed by URBasic with Modbus and urx

    """
    def __init__(self, ip):
        self.ip = ip
        self.robot = urx.Robot(ip)
        self.mb_connection = None
        self.decelleration_mbport = 135
        self.enable_realtime_mbport = 134
        self.x_mbport = 128
        self.y_mbport = 129
        self.z_mbport = 130
        self.rx_mbport = 131
        self.ry_mbport = 132
        self.rz_mbport = 133
        self.modbus_write_failure = False
        startprog="""def msg(): 
    textmsg("----------------------------") 
end"""
        self.robot.send_program(startprog)
    
    def unsigned(self, a):
        if a > 32767:
            a = a - 65535
        else:
            a = a
        return a

    def create_unsigned(self, a):
        if a <0: 
            a = a + 65535
        return a 
    
    def m_to_mmm(self, a):
        b = int(a *10000)
        print(f"input: {a}, output:{b}")
        return b

    def connect(self):
        try:
            #connect via modbus
            self.mb_connection = ModbusClient(host=self.ip, port=502, auto_open=True, debug=False)
            print("connected",c.open())
        except:
            print("Error with host or port params")

    def set_modbus_register(self, port, value, position=True, inmeters=False):
        #if inmeters:    
        if True:
            value = self.m_to_mmm(value)
        if position:
            value = self.create_unsigned(value)

        if self.mb_connection.write_single_register(port, value):
            print(f"write ok port:{port}, value: {value}")
            pass
        else:
            self.modbus_write_failure = True
            print(f"Modbus port write not successful, caution: port:{port}, value: {value}!        "*5)

    def send_realtime_stop(self):
        #may not be necessary

        pass

    def stop_realtime_control(self):
        self.set_modbus_register(self.enable_realtime_mbport, 0)

    def close(self):
        pass
    def movej(self,q, a, v, thresh = 0.0035):
        print("q",q)
        self.robot.movej(q, a, v,threshold=thresh)

    def get_actual_tcp_pose(self):
        pos = self.robot.getl()
        return pos

    def movel_waypoints(self, waypoints):
        """waypoints: List waypoint dictionaries {pose: [6d], a, v, t, r}"""
        prog = ""
        prog += "def move_linear_waypoints():\n"
        for wp in waypoints: 
            pose = wp["pose"]
            a = wp["a"]
            v = wp["v"]
            r = wp["r"]
            wp_string = f"movel(p[{pose}], a={a},v={v}, r={r})"
            prog += wpstring + "\n"
        print("moving")
        self.robot.send_program(prog)
            

    def send_interrupt(self):
        # send simple program to interrupt
        prog += """def move_linear_waypoints():
    textmsg("interrupt")
        """
        pass
    def set_realtime_decellerated_stop(self, flag):
        self.set_modbus_register(self.decelleration_mbport, int(flag)) 
        pass

    def init_realtime_control(self):
        self.set_modbus_register(self.decelleration_mbport, 0) 
        self.set_modbus_register(self.enable_realtime_mbport, 1)
        tcp_pos = self.get_actual_tcp_pose()
        print("tcp_pos", tcp_pos)
        self.set_realtime_pose(tcp_pos)

        prog = f'''def posecheck():
    if (read_port_register({self.enable_realtime_mbport})):
        textmsg(read_port_register({self.x_mbport}))
        textmsg(read_port_register({self.y_mbport}))
        textmsg(read_port_register({self.z_mbport}))
        new_pose = p[read_port_register({self.x_mbport})/10000,
                    read_port_register({self.y_mbport})/10000,
                    read_port_register({self.z_mbport})/10000,
                    read_port_register({self.rx_mbport})/10000,
                    read_port_register({self.ry_mbport})/10000,
                    read_port_register({self.rz_mbport})/10000]
        textmsg(new_pose)
    end
end
'''     
        self.robot.send_program(prog)
        
        #pass

        prog = f'''def realtime_control():
    textmsg("realtime control loop started ---------------")
    while (read_port_register({self.enable_realtime_mbport})):
        
        new_pose = p[read_port_register({self.x_mbport})/10000,
                    read_port_register({self.y_mbport})/10000,
                    read_port_register({self.z_mbport})/10000,
                    read_port_register({self.rx_mbport})/10000,
                    read_port_register({self.ry_mbport})/10000,
                    read_port_register({self.rz_mbport})/10000]
        
        decelleration_flag = read_port_register({self.decelleration_mbport})
        if decelleration_flag:
            stopj(1)
            textmsg("stop-l to prevent ramp error")
        end

           
        servoj(get_inverse_kin(new_pose), t=5.4, lookahead_time= 0.1, gain=350)            
        sync()
    end
    stopj(2)
    textmsg("realtime control loop ended---------------")
end
'''
        self.robot.send_program(prog)

    def set_realtime_pose(self,nextpose):
        self.set_modbus_register(self.x_mbport,  nextpose[0]) 
        self.set_modbus_register(self.y_mbport,  nextpose[1]) 
        self.set_modbus_register(self.z_mbport,  nextpose[2]) 
        self.set_modbus_register(self.rx_mbport, nextpose[3]) 
        self.set_modbus_register(self.ry_mbport, nextpose[4]) 
        self.set_modbus_register(self.rz_mbport, nextpose[5]) 
    
    def movel(self, pose, a, v):
        self.robot.movel(pose, a, v)
        pass
    
    def stopj(self, accel=1.5):
        self.robot.stopj(accel)
        pass
    def get_actual_joint_positions(self):
        joints= self.robot.getj()
        return joints
        pass
