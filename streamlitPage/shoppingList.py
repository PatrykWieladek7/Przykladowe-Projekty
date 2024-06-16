"Example of creating a streamlit page"
import streamlit as st

def main():

    st.title("The Shopping List")
    st.multiselect('Choose weekdays', ['Monday', 'Tuesday', 'Wednesday','Thursday','Friday','Saturday','Sunday'])
    st.date_input('Choose a date')
    st.radio('Choose a shop', ['Biedronka', 'Zabka'])
    st.write(f"**Add** or **Remove** products.")

    if 'products' not in st.session_state:
        st.session_state['products']=[]

    st.write(f"**Add** a product")
    product = str(st.text_input(' Write Name of the Product'))
    if (st.button(f"**Add**")&(product != "")):
            st.session_state.products.append(product)

    st.write(f"**Remove** a product")
    choice = str(st.text_input('Write Name of the Product'))
    if (st.button(f"**Remove**")&(choice != "")):
            new=[]
            for p in st.session_state.products:
                    if (p != choice):
                        new.append(p)
            st.session_state.products=new

    st.write(f"**THE LIST**:")
    for i, p in enumerate(st.session_state.products, start=1):
        st.write(f"{i}. {p}")

    st.slider('Rate the shopping experience', 0, 10)

if __name__ == "__main__":
    main()