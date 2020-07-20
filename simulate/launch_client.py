from minerl.env.malmo import InstanceManager

if __name__ == '__main__':
    instance = InstanceManager.Instance(9000)
    instance.launch()

    print("Running...")
    input("Enter keyboard to stop")
