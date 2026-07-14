// SPDX: MIT. Derived from Vulkan Memory Allocator, (c) 2017-2026 Advanced Micro Devices, Inc.
// Full MIT copyright/permission notice: Source/Public/Oa/Runtime/OaVma.h. See NOTICE.md.
#ifndef _OA_VMA_STL_ALLOCATOR
// STL-compatible allocator.
template<typename T>
struct OaVmaStlAllocator
{
    const VkAllocationCallbacks* const m_pCallbacks;
    typedef T value_type;

    explicit OaVmaStlAllocator(const VkAllocationCallbacks* pCallbacks) : m_pCallbacks(pCallbacks) {}
    template<typename U>
    explicit OaVmaStlAllocator(const OaVmaStlAllocator<U>& src) : m_pCallbacks(src.m_pCallbacks) {}
    OaVmaStlAllocator(const OaVmaStlAllocator&) = default;
    OaVmaStlAllocator& operator=(const OaVmaStlAllocator&) = delete;

    T* allocate(size_t n);
    void deallocate(T* p, size_t n);

    template<typename U>
    bool operator==(const OaVmaStlAllocator<U>& rhs) const
    {
        return m_pCallbacks == rhs.m_pCallbacks;
    }
    template<typename U>
    bool operator!=(const OaVmaStlAllocator<U>& rhs) const
    {
        return m_pCallbacks != rhs.m_pCallbacks;
    }
};

template<typename T>
T* OaVmaStlAllocator<T>::allocate(size_t n) { return OaVmaAllocateArray<T>(m_pCallbacks, n); }

template<typename T>
void OaVmaStlAllocator<T>::deallocate(T* p, size_t n) { OaVmaFree(m_pCallbacks, p); }
#endif // _OA_VMA_STL_ALLOCATOR

#ifndef _OA_VMA_VECTOR
/* Class with interface compatible with subset of std::vector.
T must be POD because constructors and destructors are not called and memcpy is
used for these objects. */
template<typename T, typename AllocatorT>
class OaVmaVector
{
public:
    typedef T value_type;
    typedef T* iterator;
    typedef const T* const_iterator;

    explicit OaVmaVector(const AllocatorT& allocator);
    OaVmaVector(size_t count, const AllocatorT& allocator);
    // This version of the constructor is here for compatibility with pre-C++14 std::vector.
    // value is unused.
    OaVmaVector(size_t count, const T& value, const AllocatorT& allocator) : OaVmaVector(count, allocator) {}
    OaVmaVector(const OaVmaVector<T, AllocatorT>& src);
    OaVmaVector& operator=(const OaVmaVector& rhs);
    ~OaVmaVector();

    bool empty() const { return m_Count == 0; }
    size_t size() const { return m_Count; }
    T* data() { return m_pArray; }
    T& front() { OA_VMA_HEAVY_ASSERT(m_Count > 0); return m_pArray[0]; }
    T& back() { OA_VMA_HEAVY_ASSERT(m_Count > 0); return m_pArray[m_Count - 1]; }
    const T* data() const { return m_pArray; }
    const T& front() const { OA_VMA_HEAVY_ASSERT(m_Count > 0); return m_pArray[0]; }
    const T& back() const { OA_VMA_HEAVY_ASSERT(m_Count > 0); return m_pArray[m_Count - 1]; }

    iterator begin() { return m_pArray; }
    iterator end() { return m_pArray + m_Count; }
    const_iterator cbegin() const { return m_pArray; }
    const_iterator cend() const { return m_pArray + m_Count; }
    const_iterator begin() const { return cbegin(); }
    const_iterator end() const { return cend(); }

    void pop_front() { OA_VMA_HEAVY_ASSERT(m_Count > 0); remove(0); }
    void pop_back() { OA_VMA_HEAVY_ASSERT(m_Count > 0); resize(size() - 1); }
    void push_front(const T& src) { insert(0, src); }

    void push_back(const T& src);
    void reserve(size_t newCapacity, bool freeMemory = false);
    void resize(size_t newCount);
    void clear() { resize(0); }
    void shrink_to_fit();
    void insert(size_t index, const T& src);
    void remove(size_t index);

    T& operator[](size_t index) { OA_VMA_HEAVY_ASSERT(index < m_Count); return m_pArray[index]; }
    const T& operator[](size_t index) const { OA_VMA_HEAVY_ASSERT(index < m_Count); return m_pArray[index]; }

private:
    AllocatorT m_Allocator;
    T* m_pArray;
    size_t m_Count;
    size_t m_Capacity;
};

#ifndef _OA_VMA_VECTOR_FUNCTIONS
template<typename T, typename AllocatorT>
OaVmaVector<T, AllocatorT>::~OaVmaVector() { OaVmaFree(m_Allocator.m_pCallbacks, m_pArray); }

template<typename T, typename AllocatorT>
OaVmaVector<T, AllocatorT>::OaVmaVector(const AllocatorT& allocator)
    : m_Allocator(allocator),
    m_pArray(OA_VMA_NULL),
    m_Count(0),
    m_Capacity(0) {}

template<typename T, typename AllocatorT>
OaVmaVector<T, AllocatorT>::OaVmaVector(size_t count, const AllocatorT& allocator)
    : m_Allocator(allocator),
    m_pArray(count ? (T*)OaVmaAllocateArray<T>(allocator.m_pCallbacks, count) : OA_VMA_NULL),
    m_Count(count),
    m_Capacity(count) {}

template<typename T, typename AllocatorT>
OaVmaVector<T, AllocatorT>::OaVmaVector(const OaVmaVector& src)
    : m_Allocator(src.m_Allocator),
    m_pArray(src.m_Count ? (T*)OaVmaAllocateArray<T>(src.m_Allocator.m_pCallbacks, src.m_Count) : OA_VMA_NULL),
    m_Count(src.m_Count),
    m_Capacity(src.m_Count)
{
    if (m_Count != 0)
    {
        memcpy(m_pArray, src.m_pArray, m_Count * sizeof(T));
    }
}

template<typename T, typename AllocatorT>
OaVmaVector<T, AllocatorT>& OaVmaVector<T, AllocatorT>::operator=(const OaVmaVector& rhs)
{
    if (&rhs != this)
    {
        resize(rhs.m_Count);
        if (m_Count != 0)
        {
            memcpy(m_pArray, rhs.m_pArray, m_Count * sizeof(T));
        }
    }
    return *this;
}

template<typename T, typename AllocatorT>
void OaVmaVector<T, AllocatorT>::push_back(const T& src)
{
    const size_t newIndex = size();
    resize(newIndex + 1);
    m_pArray[newIndex] = src;
}

template<typename T, typename AllocatorT>
void OaVmaVector<T, AllocatorT>::reserve(size_t newCapacity, bool freeMemory)
{
    newCapacity = OA_VMA_MAX(newCapacity, m_Count);

    if ((newCapacity < m_Capacity) && !freeMemory)
    {
        newCapacity = m_Capacity;
    }

    if (newCapacity != m_Capacity)
    {
        T* const newArray = newCapacity ? OaVmaAllocateArray<T>(m_Allocator, newCapacity) : OA_VMA_NULL;
        if (m_Count != 0)
        {
            memcpy(newArray, m_pArray, m_Count * sizeof(T));
        }
        OaVmaFree(m_Allocator.m_pCallbacks, m_pArray);
        m_Capacity = newCapacity;
        m_pArray = newArray;
    }
}

template<typename T, typename AllocatorT>
void OaVmaVector<T, AllocatorT>::resize(size_t newCount)
{
    size_t newCapacity = m_Capacity;
    if (newCount > m_Capacity)
    {
        newCapacity = OA_VMA_MAX(newCount, OA_VMA_MAX(m_Capacity * 3 / 2, (size_t)8));
    }

    if (newCapacity != m_Capacity)
    {
        OA_VMA_HEAVY_ASSERT(newCapacity > 0);
        T* const newArray = OaVmaAllocateArray<T>(m_Allocator.m_pCallbacks, newCapacity);
        const size_t elementsToCopy = OA_VMA_MIN(m_Count, newCount);
        if (elementsToCopy != 0)
        {
            memcpy(newArray, m_pArray, elementsToCopy * sizeof(T));
        }
        OaVmaFree(m_Allocator.m_pCallbacks, m_pArray);
        m_Capacity = newCapacity;
        m_pArray = newArray;
    }

    m_Count = newCount;
}

template<typename T, typename AllocatorT>
void OaVmaVector<T, AllocatorT>::shrink_to_fit()
{
    if (m_Capacity > m_Count)
    {
        T* newArray = OA_VMA_NULL;
        if (m_Count > 0)
        {
            newArray = OaVmaAllocateArray<T>(m_Allocator.m_pCallbacks, m_Count);
            memcpy(newArray, m_pArray, m_Count * sizeof(T));
        }
        OaVmaFree(m_Allocator.m_pCallbacks, m_pArray);
        m_Capacity = m_Count;
        m_pArray = newArray;
    }
}

template<typename T, typename AllocatorT>
void OaVmaVector<T, AllocatorT>::insert(size_t index, const T& src)
{
    OA_VMA_HEAVY_ASSERT(index <= m_Count);
    const size_t oldCount = size();
    resize(oldCount + 1);
    if (index < oldCount)
    {
        memmove(m_pArray + (index + 1), m_pArray + index, (oldCount - index) * sizeof(T));
    }
    m_pArray[index] = src;
}

template<typename T, typename AllocatorT>
void OaVmaVector<T, AllocatorT>::remove(size_t index)
{
    OA_VMA_HEAVY_ASSERT(index < m_Count);
    const size_t oldCount = size();
    if (index < oldCount - 1)
    {
        memmove(m_pArray + index, m_pArray + (index + 1), (oldCount - index - 1) * sizeof(T));
    }
    resize(oldCount - 1);
}
#endif // _OA_VMA_VECTOR_FUNCTIONS

namespace
{

template<typename T, typename allocatorT>
void OaVmaVectorInsert(OaVmaVector<T, allocatorT>& vec, size_t index, const T& item)
{
    vec.insert(index, item);
}

template<typename T, typename allocatorT>
void OaVmaVectorRemove(OaVmaVector<T, allocatorT>& vec, size_t index)
{
    vec.remove(index);
}

} // namespace

#endif // _OA_VMA_VECTOR

#ifndef _OA_VMA_SMALL_VECTOR
/*
This is a vector (a variable-sized array), optimized for the case when the array is small.

It contains some number of elements in-place, which allows it to avoid heap allocation
when the actual number of elements is below that threshold. This allows normal "small"
cases to be fast without losing generality for large inputs.
*/
template<typename T, typename AllocatorT, size_t N>
class OaVmaSmallVector
{
public:
    typedef T value_type;
    typedef T* iterator;

    explicit OaVmaSmallVector(const AllocatorT& allocator);
    OaVmaSmallVector(size_t count, const AllocatorT& allocator);
    template<typename SrcT, typename SrcAllocatorT, size_t SrcN>
    explicit OaVmaSmallVector(const OaVmaSmallVector<SrcT, SrcAllocatorT, SrcN>&) = delete;
    template<typename SrcT, typename SrcAllocatorT, size_t SrcN>
    OaVmaSmallVector<T, AllocatorT, N>& operator=(const OaVmaSmallVector<SrcT, SrcAllocatorT, SrcN>&) = delete;
    ~OaVmaSmallVector() = default;

    bool empty() const { return m_Count == 0; }
    size_t size() const { return m_Count; }
    T* data() { return m_Count > N ? m_DynamicArray.data() : m_StaticArray; }
    T& front() { OA_VMA_HEAVY_ASSERT(m_Count > 0); return data()[0]; }
    T& back() { OA_VMA_HEAVY_ASSERT(m_Count > 0); return data()[m_Count - 1]; }
    const T* data() const { return m_Count > N ? m_DynamicArray.data() : m_StaticArray; }
    const T& front() const { OA_VMA_HEAVY_ASSERT(m_Count > 0); return data()[0]; }
    const T& back() const { OA_VMA_HEAVY_ASSERT(m_Count > 0); return data()[m_Count - 1]; }

    iterator begin() { return data(); }
    iterator end() { return data() + m_Count; }

    void pop_front() { OA_VMA_HEAVY_ASSERT(m_Count > 0); remove(0); }
    void pop_back() { OA_VMA_HEAVY_ASSERT(m_Count > 0); resize(size() - 1); }
    void push_front(const T& src) { insert(0, src); }

    void push_back(const T& src);
    void resize(size_t newCount, bool freeMemory = false);
    void clear(bool freeMemory = false);
    void insert(size_t index, const T& src);
    void remove(size_t index);

    T& operator[](size_t index) { OA_VMA_HEAVY_ASSERT(index < m_Count); return data()[index]; }
    const T& operator[](size_t index) const { OA_VMA_HEAVY_ASSERT(index < m_Count); return data()[index]; }

private:
    size_t m_Count;
    T m_StaticArray[N]; // Used when m_Size <= N
    OaVmaVector<T, AllocatorT> m_DynamicArray; // Used when m_Size > N
};

#ifndef _OA_VMA_SMALL_VECTOR_FUNCTIONS
template<typename T, typename AllocatorT, size_t N>
OaVmaSmallVector<T, AllocatorT, N>::OaVmaSmallVector(const AllocatorT& allocator)
    : m_Count(0),
    m_DynamicArray(allocator) {}

template<typename T, typename AllocatorT, size_t N>
OaVmaSmallVector<T, AllocatorT, N>::OaVmaSmallVector(size_t count, const AllocatorT& allocator)
    : m_Count(count),
    m_DynamicArray(count > N ? count : 0, allocator) {}

template<typename T, typename AllocatorT, size_t N>
void OaVmaSmallVector<T, AllocatorT, N>::push_back(const T& src)
{
    const size_t newIndex = size();
    resize(newIndex + 1);
    data()[newIndex] = src;
}

template<typename T, typename AllocatorT, size_t N>
void OaVmaSmallVector<T, AllocatorT, N>::resize(size_t newCount, bool freeMemory)
{
    if (newCount > N && m_Count > N)
    {
        // Any direction, staying in m_DynamicArray
        m_DynamicArray.resize(newCount);
        if (freeMemory)
        {
            m_DynamicArray.shrink_to_fit();
        }
    }
    else if (newCount > N && m_Count <= N)
    {
        // Growing, moving from m_StaticArray to m_DynamicArray
        m_DynamicArray.resize(newCount);
        if (m_Count > 0)
        {
            memcpy(m_DynamicArray.data(), m_StaticArray, m_Count * sizeof(T));
        }
    }
    else if (newCount <= N && m_Count > N)
    {
        // Shrinking, moving from m_DynamicArray to m_StaticArray
        if (newCount > 0)
        {
            memcpy(m_StaticArray, m_DynamicArray.data(), newCount * sizeof(T));
        }
        m_DynamicArray.resize(0);
        if (freeMemory)
        {
            m_DynamicArray.shrink_to_fit();
        }
    }
    else
    {
        // Any direction, staying in m_StaticArray - nothing to do here
    }
    m_Count = newCount;
}

template<typename T, typename AllocatorT, size_t N>
void OaVmaSmallVector<T, AllocatorT, N>::clear(bool freeMemory)
{
    m_DynamicArray.clear();
    if (freeMemory)
    {
        m_DynamicArray.shrink_to_fit();
    }
    m_Count = 0;
}

template<typename T, typename AllocatorT, size_t N>
void OaVmaSmallVector<T, AllocatorT, N>::insert(size_t index, const T& src)
{
    OA_VMA_HEAVY_ASSERT(index <= m_Count);
    const size_t oldCount = size();
    resize(oldCount + 1);
    T* const dataPtr = data();
    if (index < oldCount)
    {
        //  I know, this could be more optimal for case where memmove can be memcpy directly from m_StaticArray to m_DynamicArray.
        memmove(dataPtr + (index + 1), dataPtr + index, (oldCount - index) * sizeof(T));
    }
    dataPtr[index] = src;
}

template<typename T, typename AllocatorT, size_t N>
void OaVmaSmallVector<T, AllocatorT, N>::remove(size_t index)
{
    OA_VMA_HEAVY_ASSERT(index < m_Count);
    const size_t oldCount = size();
    if (index < oldCount - 1)
    {
        //  I know, this could be more optimal for case where memmove can be memcpy directly from m_DynamicArray to m_StaticArray.
        T* const dataPtr = data();
        memmove(dataPtr + index, dataPtr + (index + 1), (oldCount - index - 1) * sizeof(T));
    }
    resize(oldCount - 1);
}
#endif // _OA_VMA_SMALL_VECTOR_FUNCTIONS
#endif // _OA_VMA_SMALL_VECTOR

#ifndef _OA_VMA_POOL_ALLOCATOR
/*
Allocator for objects of type T using a list of arrays (pools) to speed up
allocation. Number of elements that can be allocated is not bounded because
allocator can create multiple blocks.
*/
template<typename T>
class OaVmaPoolAllocator
{
    OA_VMA_CLASS_NO_COPY_NO_MOVE(OaVmaPoolAllocator)
public:
    OaVmaPoolAllocator(const VkAllocationCallbacks* pAllocationCallbacks, uint32_t firstBlockCapacity);
    ~OaVmaPoolAllocator();
    template<typename... Types> T* Alloc(Types&&... args);
    void Free(T* ptr);

private:
    union Item
    {
        uint32_t NextFreeIndex;
        alignas(T) char Value[sizeof(T)];
    };
    struct ItemBlock
    {
        Item* pItems;
        uint32_t Capacity;
        uint32_t FirstFreeIndex;
    };

    const VkAllocationCallbacks* m_pAllocationCallbacks;
    const uint32_t m_FirstBlockCapacity;
    OaVmaVector<ItemBlock, OaVmaStlAllocator<ItemBlock>> m_ItemBlocks;

    ItemBlock& CreateNewBlock();
};

#ifndef _OA_VMA_POOL_ALLOCATOR_FUNCTIONS
template<typename T>
OaVmaPoolAllocator<T>::OaVmaPoolAllocator(const VkAllocationCallbacks* pAllocationCallbacks, uint32_t firstBlockCapacity)
    : m_pAllocationCallbacks(pAllocationCallbacks),
    m_FirstBlockCapacity(firstBlockCapacity),
    m_ItemBlocks(OaVmaStlAllocator<ItemBlock>(pAllocationCallbacks))
{
    OA_VMA_ASSERT(m_FirstBlockCapacity > 1);
}

template<typename T>
OaVmaPoolAllocator<T>::~OaVmaPoolAllocator()
{
    for (size_t i = m_ItemBlocks.size(); i--;)
        OaVma_delete_array(m_pAllocationCallbacks, m_ItemBlocks[i].pItems, m_ItemBlocks[i].Capacity);
    m_ItemBlocks.clear();
}

template<typename T>
template<typename... Types> T* OaVmaPoolAllocator<T>::Alloc(Types&&... args)
{
    for (size_t i = m_ItemBlocks.size(); i--; )
    {
        ItemBlock& block = m_ItemBlocks[i];
        // This block has some free items: Use first one.
        if (block.FirstFreeIndex != UINT32_MAX)
        {
            Item* const pItem = &block.pItems[block.FirstFreeIndex];
            block.FirstFreeIndex = pItem->NextFreeIndex;
            T* result = (T*)&pItem->Value;
            new(result)T(std::forward<Types>(args)...); // Explicit constructor call.
            return result;
        }
    }

    // No block has free item: Create new one and use it.
    ItemBlock& newBlock = CreateNewBlock();
    Item* const pItem = &newBlock.pItems[0];
    newBlock.FirstFreeIndex = pItem->NextFreeIndex;
    T* result = (T*)&pItem->Value;
    new(result) T(std::forward<Types>(args)...); // Explicit constructor call.
    return result;
}

template<typename T>
void OaVmaPoolAllocator<T>::Free(T* ptr)
{
    // Search all memory blocks to find ptr.
    for (size_t i = m_ItemBlocks.size(); i--; )
    {
        ItemBlock& block = m_ItemBlocks[i];

        // Casting to union.
        Item* pItemPtr = OA_VMA_NULL;
        memcpy(&pItemPtr, &ptr, sizeof(pItemPtr));

        // Check if pItemPtr is in address range of this block.
        if ((pItemPtr >= block.pItems) && (pItemPtr < block.pItems + block.Capacity))
        {
            ptr->~T(); // Explicit destructor call.
            const uint32_t index = static_cast<uint32_t>(pItemPtr - block.pItems);
            pItemPtr->NextFreeIndex = block.FirstFreeIndex;
            block.FirstFreeIndex = index;
            return;
        }
    }
    OA_VMA_ASSERT(0 && "Pointer doesn't belong to this memory pool.");
}

template<typename T>
typename OaVmaPoolAllocator<T>::ItemBlock& OaVmaPoolAllocator<T>::CreateNewBlock()
{
    const uint32_t newBlockCapacity = m_ItemBlocks.empty() ?
        m_FirstBlockCapacity : m_ItemBlocks.back().Capacity * 3 / 2;

    const ItemBlock newBlock =
    {
        OaVma_new_array(m_pAllocationCallbacks, Item, newBlockCapacity),
        newBlockCapacity,
        0
    };

    m_ItemBlocks.push_back(newBlock);

    // Setup singly-linked list of all free items in this block.
    for (uint32_t i = 0; i < newBlockCapacity - 1; ++i)
        newBlock.pItems[i].NextFreeIndex = i + 1;
    newBlock.pItems[newBlockCapacity - 1].NextFreeIndex = UINT32_MAX;
    return m_ItemBlocks.back();
}
#endif // _OA_VMA_POOL_ALLOCATOR_FUNCTIONS
#endif // _OA_VMA_POOL_ALLOCATOR

#ifndef _OA_VMA_RAW_LIST
template<typename T>
struct OaVmaListItem
{
    OaVmaListItem* pPrev;
    OaVmaListItem* pNext;
    T Value;
};

// Doubly linked list.
template<typename T>
class OaVmaRawList
{
    OA_VMA_CLASS_NO_COPY_NO_MOVE(OaVmaRawList)
public:
    typedef OaVmaListItem<T> ItemType;

    explicit OaVmaRawList(const VkAllocationCallbacks* pAllocationCallbacks);
    // Intentionally not calling Clear, because that would be unnecessary
    // computations to return all items to m_ItemAllocator as free.
    ~OaVmaRawList() = default;

    size_t GetCount() const { return m_Count; }
    bool IsEmpty() const { return m_Count == 0; }

    ItemType* Front() { return m_pFront; }
    ItemType* Back() { return m_pBack; }
    const ItemType* Front() const { return m_pFront; }
    const ItemType* Back() const { return m_pBack; }

    ItemType* PushFront();
    ItemType* PushBack();
    ItemType* PushFront(const T& value);
    ItemType* PushBack(const T& value);
    void PopFront();
    void PopBack();

    // Item can be null - it means PushBack.
    ItemType* InsertBefore(ItemType* pItem);
    // Item can be null - it means PushFront.
    ItemType* InsertAfter(ItemType* pItem);
    ItemType* InsertBefore(ItemType* pItem, const T& value);
    ItemType* InsertAfter(ItemType* pItem, const T& value);

    void Clear();
    void Remove(ItemType* pItem);

private:
    const VkAllocationCallbacks* const m_pAllocationCallbacks;
    OaVmaPoolAllocator<ItemType> m_ItemAllocator;
    ItemType* m_pFront;
    ItemType* m_pBack;
    size_t m_Count;
};

#ifndef _OA_VMA_RAW_LIST_FUNCTIONS
template<typename T>
OaVmaRawList<T>::OaVmaRawList(const VkAllocationCallbacks* pAllocationCallbacks)
    : m_pAllocationCallbacks(pAllocationCallbacks),
    m_ItemAllocator(pAllocationCallbacks, 128),
    m_pFront(OA_VMA_NULL),
    m_pBack(OA_VMA_NULL),
    m_Count(0) {}

template<typename T>
OaVmaListItem<T>* OaVmaRawList<T>::PushFront()
{
    ItemType* const pNewItem = m_ItemAllocator.Alloc();
    pNewItem->pPrev = OA_VMA_NULL;
    if (IsEmpty())
    {
        pNewItem->pNext = OA_VMA_NULL;
        m_pFront = pNewItem;
        m_pBack = pNewItem;
        m_Count = 1;
    }
    else
    {
        pNewItem->pNext = m_pFront;
        m_pFront->pPrev = pNewItem;
        m_pFront = pNewItem;
        ++m_Count;
    }
    return pNewItem;
}

template<typename T>
OaVmaListItem<T>* OaVmaRawList<T>::PushBack()
{
    ItemType* const pNewItem = m_ItemAllocator.Alloc();
    pNewItem->pNext = OA_VMA_NULL;
    if(IsEmpty())
    {
        pNewItem->pPrev = OA_VMA_NULL;
        m_pFront = pNewItem;
        m_pBack = pNewItem;
        m_Count = 1;
    }
    else
    {
        pNewItem->pPrev = m_pBack;
        m_pBack->pNext = pNewItem;
        m_pBack = pNewItem;
        ++m_Count;
    }
    return pNewItem;
}

template<typename T>
OaVmaListItem<T>* OaVmaRawList<T>::PushFront(const T& value)
{
    ItemType* const pNewItem = PushFront();
    pNewItem->Value = value;
    return pNewItem;
}

template<typename T>
OaVmaListItem<T>* OaVmaRawList<T>::PushBack(const T& value)
{
    ItemType* const pNewItem = PushBack();
    pNewItem->Value = value;
    return pNewItem;
}

template<typename T>
void OaVmaRawList<T>::PopFront()
{
    OA_VMA_HEAVY_ASSERT(m_Count > 0);
    ItemType* const pFrontItem = m_pFront;
    ItemType* const pNextItem = pFrontItem->pNext;
    if (pNextItem != OA_VMA_NULL)
    {
        pNextItem->pPrev = OA_VMA_NULL;
    }
    m_pFront = pNextItem;
    m_ItemAllocator.Free(pFrontItem);
    --m_Count;
}

template<typename T>
void OaVmaRawList<T>::PopBack()
{
    OA_VMA_HEAVY_ASSERT(m_Count > 0);
    ItemType* const pBackItem = m_pBack;
    ItemType* const pPrevItem = pBackItem->pPrev;
    if(pPrevItem != OA_VMA_NULL)
    {
        pPrevItem->pNext = OA_VMA_NULL;
    }
    m_pBack = pPrevItem;
    m_ItemAllocator.Free(pBackItem);
    --m_Count;
}

template<typename T>
void OaVmaRawList<T>::Clear()
{
    if (!IsEmpty())
    {
        ItemType* pItem = m_pBack;
        while (pItem != OA_VMA_NULL)
        {
            ItemType* const pPrevItem = pItem->pPrev;
            m_ItemAllocator.Free(pItem);
            pItem = pPrevItem;
        }
        m_pFront = OA_VMA_NULL;
        m_pBack = OA_VMA_NULL;
        m_Count = 0;
    }
}

template<typename T>
void OaVmaRawList<T>::Remove(ItemType* pItem)
{
    OA_VMA_HEAVY_ASSERT(pItem != OA_VMA_NULL);
    OA_VMA_HEAVY_ASSERT(m_Count > 0);

    if(pItem->pPrev != OA_VMA_NULL)
    {
        pItem->pPrev->pNext = pItem->pNext;
    }
    else
    {
        OA_VMA_HEAVY_ASSERT(m_pFront == pItem);
        m_pFront = pItem->pNext;
    }

    if(pItem->pNext != OA_VMA_NULL)
    {
        pItem->pNext->pPrev = pItem->pPrev;
    }
    else
    {
        OA_VMA_HEAVY_ASSERT(m_pBack == pItem);
        m_pBack = pItem->pPrev;
    }

    m_ItemAllocator.Free(pItem);
    --m_Count;
}

template<typename T>
OaVmaListItem<T>* OaVmaRawList<T>::InsertBefore(ItemType* pItem)
{
    if(pItem != OA_VMA_NULL)
    {
        ItemType* const prevItem = pItem->pPrev;
        ItemType* const newItem = m_ItemAllocator.Alloc();
        newItem->pPrev = prevItem;
        newItem->pNext = pItem;
        pItem->pPrev = newItem;
        if(prevItem != OA_VMA_NULL)
        {
            prevItem->pNext = newItem;
        }
        else
        {
            OA_VMA_HEAVY_ASSERT(m_pFront == pItem);
            m_pFront = newItem;
        }
        ++m_Count;
        return newItem;
    }
    return PushBack();
}

template<typename T>
OaVmaListItem<T>* OaVmaRawList<T>::InsertAfter(ItemType* pItem)
{
    if(pItem != OA_VMA_NULL)
    {
        ItemType* const nextItem = pItem->pNext;
        ItemType* const newItem = m_ItemAllocator.Alloc();
        newItem->pNext = nextItem;
        newItem->pPrev = pItem;
        pItem->pNext = newItem;
        if(nextItem != OA_VMA_NULL)
        {
            nextItem->pPrev = newItem;
        }
        else
        {
            OA_VMA_HEAVY_ASSERT(m_pBack == pItem);
            m_pBack = newItem;
        }
        ++m_Count;
        return newItem;
    }
    return PushFront();
}

template<typename T>
OaVmaListItem<T>* OaVmaRawList<T>::InsertBefore(ItemType* pItem, const T& value)
{
    ItemType* const newItem = InsertBefore(pItem);
    newItem->Value = value;
    return newItem;
}

template<typename T>
OaVmaListItem<T>* OaVmaRawList<T>::InsertAfter(ItemType* pItem, const T& value)
{
    ItemType* const newItem = InsertAfter(pItem);
    newItem->Value = value;
    return newItem;
}
#endif // _OA_VMA_RAW_LIST_FUNCTIONS
#endif // _OA_VMA_RAW_LIST

#ifndef _OA_VMA_LIST
template<typename T, typename AllocatorT>
class OaVmaList
{
    OA_VMA_CLASS_NO_COPY_NO_MOVE(OaVmaList)
public:
    class reverse_iterator;
    class const_iterator;
    class const_reverse_iterator;

    class iterator
    {
        friend class const_iterator;
        friend class OaVmaList<T, AllocatorT>;
    public:
        iterator() :  m_pList(OA_VMA_NULL), m_pItem(OA_VMA_NULL) {}
        explicit iterator(const reverse_iterator& src) : m_pList(src.m_pList), m_pItem(src.m_pItem) {}

        T& operator*() const { OA_VMA_HEAVY_ASSERT(m_pItem != OA_VMA_NULL); return m_pItem->Value; }
        T* operator->() const { OA_VMA_HEAVY_ASSERT(m_pItem != OA_VMA_NULL); return &m_pItem->Value; }

        bool operator==(const iterator& rhs) const { OA_VMA_HEAVY_ASSERT(m_pList == rhs.m_pList); return m_pItem == rhs.m_pItem; }
        bool operator!=(const iterator& rhs) const { OA_VMA_HEAVY_ASSERT(m_pList == rhs.m_pList); return m_pItem != rhs.m_pItem; }

        const iterator operator++(int) { iterator result = *this; ++*this; return result; }
        const iterator operator--(int) { iterator result = *this; --*this; return result; }

        iterator& operator++() { OA_VMA_HEAVY_ASSERT(m_pItem != OA_VMA_NULL); m_pItem = m_pItem->pNext; return *this; }
        iterator& operator--();

    private:
        OaVmaRawList<T>* m_pList;
        OaVmaListItem<T>* m_pItem;

        iterator(OaVmaRawList<T>* pList, OaVmaListItem<T>* pItem) : m_pList(pList),  m_pItem(pItem) {}
    };
    class reverse_iterator
    {
        friend class const_reverse_iterator;
        friend class OaVmaList<T, AllocatorT>;
    public:
        reverse_iterator() : m_pList(OA_VMA_NULL), m_pItem(OA_VMA_NULL) {}
        explicit reverse_iterator(const iterator& src) : m_pList(src.m_pList), m_pItem(src.m_pItem) {}

        T& operator*() const { OA_VMA_HEAVY_ASSERT(m_pItem != OA_VMA_NULL); return m_pItem->Value; }
        T* operator->() const { OA_VMA_HEAVY_ASSERT(m_pItem != OA_VMA_NULL); return &m_pItem->Value; }

        bool operator==(const reverse_iterator& rhs) const { OA_VMA_HEAVY_ASSERT(m_pList == rhs.m_pList); return m_pItem == rhs.m_pItem; }
        bool operator!=(const reverse_iterator& rhs) const { OA_VMA_HEAVY_ASSERT(m_pList == rhs.m_pList); return m_pItem != rhs.m_pItem; }

        const reverse_iterator operator++(int) { reverse_iterator result = *this; ++* this; return result; }
        const reverse_iterator operator--(int) { reverse_iterator result = *this; --* this; return result; }

        reverse_iterator& operator++() { OA_VMA_HEAVY_ASSERT(m_pItem != OA_VMA_NULL); m_pItem = m_pItem->pPrev; return *this; }
        reverse_iterator& operator--();

    private:
        OaVmaRawList<T>* m_pList;
        OaVmaListItem<T>* m_pItem;

        reverse_iterator(OaVmaRawList<T>* pList, OaVmaListItem<T>* pItem) : m_pList(pList),  m_pItem(pItem) {}
    };
    class const_iterator
    {
        friend class OaVmaList<T, AllocatorT>;
    public:
        const_iterator() : m_pList(OA_VMA_NULL), m_pItem(OA_VMA_NULL) {}
        explicit const_iterator(const iterator& src) : m_pList(src.m_pList), m_pItem(src.m_pItem) {}
        explicit const_iterator(const reverse_iterator& src) : m_pList(src.m_pList), m_pItem(src.m_pItem) {}

        iterator drop_const() { return { const_cast<OaVmaRawList<T>*>(m_pList), const_cast<OaVmaListItem<T>*>(m_pItem) }; }

        const T& operator*() const { OA_VMA_HEAVY_ASSERT(m_pItem != OA_VMA_NULL); return m_pItem->Value; }
        const T* operator->() const { OA_VMA_HEAVY_ASSERT(m_pItem != OA_VMA_NULL); return &m_pItem->Value; }

        bool operator==(const const_iterator& rhs) const { OA_VMA_HEAVY_ASSERT(m_pList == rhs.m_pList); return m_pItem == rhs.m_pItem; }
        bool operator!=(const const_iterator& rhs) const { OA_VMA_HEAVY_ASSERT(m_pList == rhs.m_pList); return m_pItem != rhs.m_pItem; }

        const const_iterator operator++(int) { const_iterator result = *this; ++* this; return result; }
        const const_iterator operator--(int) { const_iterator result = *this; --* this; return result; }

        const_iterator& operator++() { OA_VMA_HEAVY_ASSERT(m_pItem != OA_VMA_NULL); m_pItem = m_pItem->pNext; return *this; }
        const_iterator& operator--();

    private:
        const OaVmaRawList<T>* m_pList;
        const OaVmaListItem<T>* m_pItem;

        const_iterator(const OaVmaRawList<T>* pList, const OaVmaListItem<T>* pItem) : m_pList(pList), m_pItem(pItem) {}
    };
    class const_reverse_iterator
    {
        friend class OaVmaList<T, AllocatorT>;
    public:
        const_reverse_iterator() : m_pList(OA_VMA_NULL), m_pItem(OA_VMA_NULL) {}
        explicit const_reverse_iterator(const reverse_iterator& src) : m_pList(src.m_pList), m_pItem(src.m_pItem) {}
        explicit const_reverse_iterator(const iterator& src) : m_pList(src.m_pList), m_pItem(src.m_pItem) {}

        reverse_iterator drop_const() { return { const_cast<OaVmaRawList<T>*>(m_pList), const_cast<OaVmaListItem<T>*>(m_pItem) }; }

        const T& operator*() const { OA_VMA_HEAVY_ASSERT(m_pItem != OA_VMA_NULL); return m_pItem->Value; }
        const T* operator->() const { OA_VMA_HEAVY_ASSERT(m_pItem != OA_VMA_NULL); return &m_pItem->Value; }

        bool operator==(const const_reverse_iterator& rhs) const { OA_VMA_HEAVY_ASSERT(m_pList == rhs.m_pList); return m_pItem == rhs.m_pItem; }
        bool operator!=(const const_reverse_iterator& rhs) const { OA_VMA_HEAVY_ASSERT(m_pList == rhs.m_pList); return m_pItem != rhs.m_pItem; }

        const const_reverse_iterator operator++(int) { const_reverse_iterator result = *this; ++* this; return result; }
        const const_reverse_iterator operator--(int) { const_reverse_iterator result = *this; --* this; return result; }

        const_reverse_iterator& operator++() { OA_VMA_HEAVY_ASSERT(m_pItem != OA_VMA_NULL); m_pItem = m_pItem->pPrev; return *this; }
        const_reverse_iterator& operator--();

    private:
        const OaVmaRawList<T>* m_pList;
        const OaVmaListItem<T>* m_pItem;

        const_reverse_iterator(const OaVmaRawList<T>* pList, const OaVmaListItem<T>* pItem) : m_pList(pList), m_pItem(pItem) {}
    };

    explicit OaVmaList(const AllocatorT& allocator) : m_RawList(allocator.m_pCallbacks) {}

    bool empty() const { return m_RawList.IsEmpty(); }
    size_t size() const { return m_RawList.GetCount(); }

    iterator begin() { return iterator(&m_RawList, m_RawList.Front()); }
    iterator end() { return iterator(&m_RawList, OA_VMA_NULL); }

    const_iterator cbegin() const { return const_iterator(&m_RawList, m_RawList.Front()); }
    const_iterator cend() const { return const_iterator(&m_RawList, OA_VMA_NULL); }

    const_iterator begin() const { return cbegin(); }
    const_iterator end() const { return cend(); }

    reverse_iterator rbegin() { return reverse_iterator(&m_RawList, m_RawList.Back()); }
    reverse_iterator rend() { return reverse_iterator(&m_RawList, OA_VMA_NULL); }

    const_reverse_iterator crbegin() const { return const_reverse_iterator(&m_RawList, m_RawList.Back()); }
    const_reverse_iterator crend() const { return const_reverse_iterator(&m_RawList, OA_VMA_NULL); }

    const_reverse_iterator rbegin() const { return crbegin(); }
    const_reverse_iterator rend() const { return crend(); }

    void push_back(const T& value) { m_RawList.PushBack(value); }
    iterator insert(iterator it, const T& value) { return iterator(&m_RawList, m_RawList.InsertBefore(it.m_pItem, value)); }

    void clear() { m_RawList.Clear(); }
    void erase(iterator it) { m_RawList.Remove(it.m_pItem); }

private:
    OaVmaRawList<T> m_RawList;
};

#ifndef _OA_VMA_LIST_FUNCTIONS
template<typename T, typename AllocatorT>
typename OaVmaList<T, AllocatorT>::iterator& OaVmaList<T, AllocatorT>::iterator::operator--()
{
    if (m_pItem != OA_VMA_NULL)
    {
        m_pItem = m_pItem->pPrev;
    }
    else
    {
        OA_VMA_HEAVY_ASSERT(!m_pList->IsEmpty());
        m_pItem = m_pList->Back();
    }
    return *this;
}

template<typename T, typename AllocatorT>
typename OaVmaList<T, AllocatorT>::reverse_iterator& OaVmaList<T, AllocatorT>::reverse_iterator::operator--()
{
    if (m_pItem != OA_VMA_NULL)
    {
        m_pItem = m_pItem->pNext;
    }
    else
    {
        OA_VMA_HEAVY_ASSERT(!m_pList->IsEmpty());
        m_pItem = m_pList->Front();
    }
    return *this;
}

template<typename T, typename AllocatorT>
typename OaVmaList<T, AllocatorT>::const_iterator& OaVmaList<T, AllocatorT>::const_iterator::operator--()
{
    if (m_pItem != OA_VMA_NULL)
    {
        m_pItem = m_pItem->pPrev;
    }
    else
    {
        OA_VMA_HEAVY_ASSERT(!m_pList->IsEmpty());
        m_pItem = m_pList->Back();
    }
    return *this;
}

template<typename T, typename AllocatorT>
typename OaVmaList<T, AllocatorT>::const_reverse_iterator& OaVmaList<T, AllocatorT>::const_reverse_iterator::operator--()
{
    if (m_pItem != OA_VMA_NULL)
    {
        m_pItem = m_pItem->pNext;
    }
    else
    {
        OA_VMA_HEAVY_ASSERT(!m_pList->IsEmpty());
        m_pItem = m_pList->Back();
    }
    return *this;
}
#endif // _OA_VMA_LIST_FUNCTIONS
#endif // _OA_VMA_LIST

#ifndef _OA_VMA_INTRUSIVE_LINKED_LIST
/*
Expected interface of ItemTypeTraits:
struct MyItemTypeTraits
{
    typedef MyItem ItemType;
    static ItemType* GetPrev(const ItemType* item) { return item->myPrevPtr; }
    static ItemType* GetNext(const ItemType* item) { return item->myNextPtr; }
    static ItemType*& AccessPrev(ItemType* item) { return item->myPrevPtr; }
    static ItemType*& AccessNext(ItemType* item) { return item->myNextPtr; }
};
*/
template<typename ItemTypeTraits>
class OaVmaIntrusiveLinkedList
{
public:
    typedef typename ItemTypeTraits::ItemType ItemType;
    static ItemType* GetPrev(const ItemType* item) { return ItemTypeTraits::GetPrev(item); }
    static ItemType* GetNext(const ItemType* item) { return ItemTypeTraits::GetNext(item); }

    // Movable, not copyable.
    OaVmaIntrusiveLinkedList() = default;
    OaVmaIntrusiveLinkedList(OaVmaIntrusiveLinkedList && src) noexcept;
    OaVmaIntrusiveLinkedList(const OaVmaIntrusiveLinkedList&) = delete;
    OaVmaIntrusiveLinkedList& operator=(OaVmaIntrusiveLinkedList&& src) noexcept;
    OaVmaIntrusiveLinkedList& operator=(const OaVmaIntrusiveLinkedList&) = delete;
    ~OaVmaIntrusiveLinkedList() { OA_VMA_HEAVY_ASSERT(IsEmpty()); }

    size_t GetCount() const { return m_Count; }
    bool IsEmpty() const { return m_Count == 0; }
    ItemType* Front() { return m_Front; }
    ItemType* Back() { return m_Back; }
    const ItemType* Front() const { return m_Front; }
    const ItemType* Back() const { return m_Back; }

    void PushBack(ItemType* item);
    void PushFront(ItemType* item);
    ItemType* PopBack();
    ItemType* PopFront();

    // MyItem can be null - it means PushBack.
    void InsertBefore(ItemType* existingItem, ItemType* newItem);
    // MyItem can be null - it means PushFront.
    void InsertAfter(ItemType* existingItem, ItemType* newItem);
    void Remove(ItemType* item);
    void RemoveAll();

private:
    ItemType* m_Front = OA_VMA_NULL;
    ItemType* m_Back = OA_VMA_NULL;
    size_t m_Count = 0;
};

#ifndef _OA_VMA_INTRUSIVE_LINKED_LIST_FUNCTIONS
template<typename ItemTypeTraits>
OaVmaIntrusiveLinkedList<ItemTypeTraits>::OaVmaIntrusiveLinkedList(OaVmaIntrusiveLinkedList&& src) noexcept
    : m_Front(src.m_Front), m_Back(src.m_Back), m_Count(src.m_Count)
{
    src.m_Front = src.m_Back = OA_VMA_NULL;
    src.m_Count = 0;
}

template<typename ItemTypeTraits>
OaVmaIntrusiveLinkedList<ItemTypeTraits>& OaVmaIntrusiveLinkedList<ItemTypeTraits>::operator=(OaVmaIntrusiveLinkedList&& src) noexcept
{
    if (&src != this)
    {
        OA_VMA_HEAVY_ASSERT(IsEmpty());
        m_Front = src.m_Front;
        m_Back = src.m_Back;
        m_Count = src.m_Count;
        src.m_Front = src.m_Back = OA_VMA_NULL;
        src.m_Count = 0;
    }
    return *this;
}

template<typename ItemTypeTraits>
void OaVmaIntrusiveLinkedList<ItemTypeTraits>::PushBack(ItemType* item)
{
    OA_VMA_HEAVY_ASSERT(ItemTypeTraits::GetPrev(item) == OA_VMA_NULL && ItemTypeTraits::GetNext(item) == OA_VMA_NULL);
    if (IsEmpty())
    {
        m_Front = item;
        m_Back = item;
        m_Count = 1;
    }
    else
    {
        ItemTypeTraits::AccessPrev(item) = m_Back;
        ItemTypeTraits::AccessNext(m_Back) = item;
        m_Back = item;
        ++m_Count;
    }
}

template<typename ItemTypeTraits>
void OaVmaIntrusiveLinkedList<ItemTypeTraits>::PushFront(ItemType* item)
{
    OA_VMA_HEAVY_ASSERT(ItemTypeTraits::GetPrev(item) == OA_VMA_NULL && ItemTypeTraits::GetNext(item) == OA_VMA_NULL);
    if (IsEmpty())
    {
        m_Front = item;
        m_Back = item;
        m_Count = 1;
    }
    else
    {
        ItemTypeTraits::AccessNext(item) = m_Front;
        ItemTypeTraits::AccessPrev(m_Front) = item;
        m_Front = item;
        ++m_Count;
    }
}

template<typename ItemTypeTraits>
typename OaVmaIntrusiveLinkedList<ItemTypeTraits>::ItemType* OaVmaIntrusiveLinkedList<ItemTypeTraits>::PopBack()
{
    OA_VMA_HEAVY_ASSERT(m_Count > 0);
    ItemType* const backItem = m_Back;
    ItemType* const prevItem = ItemTypeTraits::GetPrev(backItem);
    if (prevItem != OA_VMA_NULL)
    {
        ItemTypeTraits::AccessNext(prevItem) = OA_VMA_NULL;
    }
    m_Back = prevItem;
    --m_Count;
    ItemTypeTraits::AccessPrev(backItem) = OA_VMA_NULL;
    ItemTypeTraits::AccessNext(backItem) = OA_VMA_NULL;
    return backItem;
}

template<typename ItemTypeTraits>
typename OaVmaIntrusiveLinkedList<ItemTypeTraits>::ItemType* OaVmaIntrusiveLinkedList<ItemTypeTraits>::PopFront()
{
    OA_VMA_HEAVY_ASSERT(m_Count > 0);
    ItemType* const frontItem = m_Front;
    ItemType* const nextItem = ItemTypeTraits::GetNext(frontItem);
    if (nextItem != OA_VMA_NULL)
    {
        ItemTypeTraits::AccessPrev(nextItem) = OA_VMA_NULL;
    }
    m_Front = nextItem;
    --m_Count;
    ItemTypeTraits::AccessPrev(frontItem) = OA_VMA_NULL;
    ItemTypeTraits::AccessNext(frontItem) = OA_VMA_NULL;
    return frontItem;
}

template<typename ItemTypeTraits>
void OaVmaIntrusiveLinkedList<ItemTypeTraits>::InsertBefore(ItemType* existingItem, ItemType* newItem)
{
    OA_VMA_HEAVY_ASSERT(newItem != OA_VMA_NULL && ItemTypeTraits::GetPrev(newItem) == OA_VMA_NULL && ItemTypeTraits::GetNext(newItem) == OA_VMA_NULL);
    if (existingItem != OA_VMA_NULL)
    {
        ItemType* const prevItem = ItemTypeTraits::GetPrev(existingItem);
        ItemTypeTraits::AccessPrev(newItem) = prevItem;
        ItemTypeTraits::AccessNext(newItem) = existingItem;
        ItemTypeTraits::AccessPrev(existingItem) = newItem;
        if (prevItem != OA_VMA_NULL)
        {
            ItemTypeTraits::AccessNext(prevItem) = newItem;
        }
        else
        {
            OA_VMA_HEAVY_ASSERT(m_Front == existingItem);
            m_Front = newItem;
        }
        ++m_Count;
    }
    else
        PushBack(newItem);
}

template<typename ItemTypeTraits>
void OaVmaIntrusiveLinkedList<ItemTypeTraits>::InsertAfter(ItemType* existingItem, ItemType* newItem)
{
    OA_VMA_HEAVY_ASSERT(newItem != OA_VMA_NULL && ItemTypeTraits::GetPrev(newItem) == OA_VMA_NULL && ItemTypeTraits::GetNext(newItem) == OA_VMA_NULL);
    if (existingItem != OA_VMA_NULL)
    {
        ItemType* const nextItem = ItemTypeTraits::GetNext(existingItem);
        ItemTypeTraits::AccessNext(newItem) = nextItem;
        ItemTypeTraits::AccessPrev(newItem) = existingItem;
        ItemTypeTraits::AccessNext(existingItem) = newItem;
        if (nextItem != OA_VMA_NULL)
        {
            ItemTypeTraits::AccessPrev(nextItem) = newItem;
        }
        else
        {
            OA_VMA_HEAVY_ASSERT(m_Back == existingItem);
            m_Back = newItem;
        }
        ++m_Count;
    }
    else
        return PushFront(newItem);
}

template<typename ItemTypeTraits>
void OaVmaIntrusiveLinkedList<ItemTypeTraits>::Remove(ItemType* item)
{
    OA_VMA_HEAVY_ASSERT(item != OA_VMA_NULL && m_Count > 0);
    if (ItemTypeTraits::GetPrev(item) != OA_VMA_NULL)
    {
        ItemTypeTraits::AccessNext(ItemTypeTraits::AccessPrev(item)) = ItemTypeTraits::GetNext(item);
    }
    else
    {
        OA_VMA_HEAVY_ASSERT(m_Front == item);
        m_Front = ItemTypeTraits::GetNext(item);
    }

    if (ItemTypeTraits::GetNext(item) != OA_VMA_NULL)
    {
        ItemTypeTraits::AccessPrev(ItemTypeTraits::AccessNext(item)) = ItemTypeTraits::GetPrev(item);
    }
    else
    {
        OA_VMA_HEAVY_ASSERT(m_Back == item);
        m_Back = ItemTypeTraits::GetPrev(item);
    }
    ItemTypeTraits::AccessPrev(item) = OA_VMA_NULL;
    ItemTypeTraits::AccessNext(item) = OA_VMA_NULL;
    --m_Count;
}

template<typename ItemTypeTraits>
void OaVmaIntrusiveLinkedList<ItemTypeTraits>::RemoveAll()
{
    if (!IsEmpty())
    {
        ItemType* item = m_Back;
        while (item != OA_VMA_NULL)
        {
            ItemType* const prevItem = ItemTypeTraits::AccessPrev(item);
            ItemTypeTraits::AccessPrev(item) = OA_VMA_NULL;
            ItemTypeTraits::AccessNext(item) = OA_VMA_NULL;
            item = prevItem;
        }
        m_Front = OA_VMA_NULL;
        m_Back = OA_VMA_NULL;
        m_Count = 0;
    }
}
#endif // _OA_VMA_INTRUSIVE_LINKED_LIST_FUNCTIONS
#endif // _OA_VMA_INTRUSIVE_LINKED_LIST

#if !defined(_OA_VMA_STRING_BUILDER) && OA_VMA_STATS_STRING_ENABLED
class OaVmaStringBuilder
{
public:
    explicit OaVmaStringBuilder(const VkAllocationCallbacks* allocationCallbacks) : m_Data(OaVmaStlAllocator<char>(allocationCallbacks)) {}
    ~OaVmaStringBuilder() = default;

    size_t GetLength() const { return m_Data.size(); }
    // Returned string is not null-terminated!
    const char* GetData() const { return m_Data.data(); }
    void AddNewLine() { Add('\n'); }
    void Add(char ch) { m_Data.push_back(ch); }

    void Add(const char* pStr);
    void AddNumber(uint32_t num);
    void AddNumber(uint64_t num);
    void AddPointer(const void* ptr);

private:
    OaVmaVector<char, OaVmaStlAllocator<char>> m_Data;
};

#ifndef _OA_VMA_STRING_BUILDER_FUNCTIONS
void OaVmaStringBuilder::Add(const char* pStr)
{
    const size_t strLen = strlen(pStr);
    if (strLen > 0)
    {
        const size_t oldCount = m_Data.size();
        m_Data.resize(oldCount + strLen);
        memcpy(m_Data.data() + oldCount, pStr, strLen);
    }
}

void OaVmaStringBuilder::AddNumber(uint32_t num)
{
    char buf[11];
    buf[10] = '\0';
    char* p = &buf[10];
    do
    {
        *--p = '0' + (char)(num % 10);
        num /= 10;
    } while (num);
    Add(p);
}

void OaVmaStringBuilder::AddNumber(uint64_t num)
{
    char buf[21];
    buf[20] = '\0';
    char* p = &buf[20];
    do
    {
        *--p = '0' + (char)(num % 10);
        num /= 10;
    } while (num);
    Add(p);
}

void OaVmaStringBuilder::AddPointer(const void* ptr)
{
    char buf[21];
    OaVmaPtrToStr(buf, sizeof(buf), ptr);
    Add(buf);
}
#endif //_OA_VMA_STRING_BUILDER_FUNCTIONS
#endif // _OA_VMA_STRING_BUILDER

#if !defined(_OA_VMA_JSON_WRITER) && OA_VMA_STATS_STRING_ENABLED
/*
Allows to conveniently build a correct JSON document to be written to the
OaVmaStringBuilder passed to the constructor.
*/
class OaVmaJsonWriter
{
    OA_VMA_CLASS_NO_COPY_NO_MOVE(OaVmaJsonWriter)
public:
    // sb - string builder to write the document to. Must remain alive for the whole lifetime of this object.
    OaVmaJsonWriter(const VkAllocationCallbacks* pAllocationCallbacks, OaVmaStringBuilder& sb);
    ~OaVmaJsonWriter();

    // Begins object by writing "{".
    // Inside an object, you must call pairs of WriteString and a value, e.g.:
    // j.BeginObject(true); j.WriteString("A"); j.WriteNumber(1); j.WriteString("B"); j.WriteNumber(2); j.EndObject();
    // Will write: { "A": 1, "B": 2 }
    void BeginObject(bool singleLine = false);
    // Ends object by writing "}".
    void EndObject();

    // Begins array by writing "[".
    // Inside an array, you can write a sequence of any values.
    void BeginArray(bool singleLine = false);
    // Ends array by writing "[".
    void EndArray();

    // Writes a string value inside "".
    // pStr can contain any ANSI characters, including '"', new line etc. - they will be properly escaped.
    void WriteString(const char* pStr);

    // Begins writing a string value.
    // Call BeginString, ContinueString, ContinueString, ..., EndString instead of
    // WriteString to conveniently build the string content incrementally, made of
    // parts including numbers.
    void BeginString(const char* pStr = OA_VMA_NULL);
    // Posts next part of an open string.
    void ContinueString(const char* pStr);
    // Posts next part of an open string. The number is converted to decimal characters.
    void ContinueString(uint32_t n);
    void ContinueString(uint64_t n);
    // Posts next part of an open string. Pointer value is converted to characters
    // using "%p" formatting - shown as hexadecimal number, e.g.: 000000081276Ad00
    void ContinueString_Pointer(const void* ptr);
    // Ends writing a string value by writing '"'.
    void EndString(const char* pStr = OA_VMA_NULL);

    // Writes a number value.
    void WriteNumber(uint32_t n);
    void WriteNumber(uint64_t n);
    // Writes a boolean value - false or true.
    void WriteBool(bool b);
    // Writes a null value.
    void WriteNull();

private:
    enum COLLECTION_TYPE
    {
        COLLECTION_TYPE_OBJECT,
        COLLECTION_TYPE_ARRAY,
    };
    struct StackItem
    {
        COLLECTION_TYPE type;
        uint32_t valueCount;
        bool singleLineMode;
    };

    static const char* const INDENT;

    OaVmaStringBuilder& m_SB;
    OaVmaVector< StackItem, OaVmaStlAllocator<StackItem> > m_Stack;
    bool m_InsideString;

    void BeginValue(bool isString);
    void WriteIndent(bool oneLess = false);
};
const char* const OaVmaJsonWriter::INDENT = "  ";

#ifndef _OA_VMA_JSON_WRITER_FUNCTIONS
OaVmaJsonWriter::OaVmaJsonWriter(const VkAllocationCallbacks* pAllocationCallbacks, OaVmaStringBuilder& sb)
    : m_SB(sb),
    m_Stack(OaVmaStlAllocator<StackItem>(pAllocationCallbacks)),
    m_InsideString(false) {}

OaVmaJsonWriter::~OaVmaJsonWriter()
{
    OA_VMA_ASSERT(!m_InsideString);
    OA_VMA_ASSERT(m_Stack.empty());
}

void OaVmaJsonWriter::BeginObject(bool singleLine)
{
    OA_VMA_ASSERT(!m_InsideString);

    BeginValue(false);
    m_SB.Add('{');

    StackItem item;
    item.type = COLLECTION_TYPE_OBJECT;
    item.valueCount = 0;
    item.singleLineMode = singleLine;
    m_Stack.push_back(item);
}

void OaVmaJsonWriter::EndObject()
{
    OA_VMA_ASSERT(!m_InsideString);

    WriteIndent(true);
    m_SB.Add('}');

    OA_VMA_ASSERT(!m_Stack.empty() && m_Stack.back().type == COLLECTION_TYPE_OBJECT);
    m_Stack.pop_back();
}

void OaVmaJsonWriter::BeginArray(bool singleLine)
{
    OA_VMA_ASSERT(!m_InsideString);

    BeginValue(false);
    m_SB.Add('[');

    StackItem item;
    item.type = COLLECTION_TYPE_ARRAY;
    item.valueCount = 0;
    item.singleLineMode = singleLine;
    m_Stack.push_back(item);
}

void OaVmaJsonWriter::EndArray()
{
    OA_VMA_ASSERT(!m_InsideString);

    WriteIndent(true);
    m_SB.Add(']');

    OA_VMA_ASSERT(!m_Stack.empty() && m_Stack.back().type == COLLECTION_TYPE_ARRAY);
    m_Stack.pop_back();
}

void OaVmaJsonWriter::WriteString(const char* pStr)
{
    BeginString(pStr);
    EndString();
}

void OaVmaJsonWriter::BeginString(const char* pStr)
{
    OA_VMA_ASSERT(!m_InsideString);

    BeginValue(true);
    m_SB.Add('"');
    m_InsideString = true;
    if (pStr != OA_VMA_NULL && pStr[0] != '\0')
    {
        ContinueString(pStr);
    }
}

void OaVmaJsonWriter::ContinueString(const char* pStr)
{
    OA_VMA_ASSERT(m_InsideString);

    const size_t strLen = strlen(pStr);
    for (size_t i = 0; i < strLen; ++i)
    {
        char ch = pStr[i];
        if (ch == '\\')
        {
            m_SB.Add("\\\\");
        }
        else if (ch == '"')
        {
            m_SB.Add("\\\"");
        }
        else if ((uint8_t)ch >= 32)
        {
            m_SB.Add(ch);
        }
        else switch (ch)
        {
        case '\b':
            m_SB.Add("\\b");
            break;
        case '\f':
            m_SB.Add("\\f");
            break;
        case '\n':
            m_SB.Add("\\n");
            break;
        case '\r':
            m_SB.Add("\\r");
            break;
        case '\t':
            m_SB.Add("\\t");
            break;
        default:
            OA_VMA_ASSERT(0 && "Character not currently supported.");
        }
    }
}

void OaVmaJsonWriter::ContinueString(uint32_t n)
{
    OA_VMA_ASSERT(m_InsideString);
    m_SB.AddNumber(n);
}

void OaVmaJsonWriter::ContinueString(uint64_t n)
{
    OA_VMA_ASSERT(m_InsideString);
    m_SB.AddNumber(n);
}

void OaVmaJsonWriter::ContinueString_Pointer(const void* ptr)
{
    OA_VMA_ASSERT(m_InsideString);
    m_SB.AddPointer(ptr);
}

void OaVmaJsonWriter::EndString(const char* pStr)
{
    OA_VMA_ASSERT(m_InsideString);
    if (pStr != OA_VMA_NULL && pStr[0] != '\0')
    {
        ContinueString(pStr);
    }
    m_SB.Add('"');
    m_InsideString = false;
}

void OaVmaJsonWriter::WriteNumber(uint32_t n)
{
    OA_VMA_ASSERT(!m_InsideString);
    BeginValue(false);
    m_SB.AddNumber(n);
}

void OaVmaJsonWriter::WriteNumber(uint64_t n)
{
    OA_VMA_ASSERT(!m_InsideString);
    BeginValue(false);
    m_SB.AddNumber(n);
}

void OaVmaJsonWriter::WriteBool(bool b)
{
    OA_VMA_ASSERT(!m_InsideString);
    BeginValue(false);
    m_SB.Add(b ? "true" : "false");
}

void OaVmaJsonWriter::WriteNull()
{
    OA_VMA_ASSERT(!m_InsideString);
    BeginValue(false);
    m_SB.Add("null");
}

void OaVmaJsonWriter::BeginValue(bool isString)
{
    if (!m_Stack.empty())
    {
        StackItem& currItem = m_Stack.back();
        if (currItem.type == COLLECTION_TYPE_OBJECT &&
            currItem.valueCount % 2 == 0)
        {
            OA_VMA_ASSERT(isString);
        }

        if (currItem.type == COLLECTION_TYPE_OBJECT &&
            currItem.valueCount % 2 != 0)
        {
            m_SB.Add(": ");
        }
        else if (currItem.valueCount > 0)
        {
            m_SB.Add(", ");
            WriteIndent();
        }
        else
        {
            WriteIndent();
        }
        ++currItem.valueCount;
    }
}

void OaVmaJsonWriter::WriteIndent(bool oneLess)
{
    if (!m_Stack.empty() && !m_Stack.back().singleLineMode)
    {
        m_SB.AddNewLine();

        size_t count = m_Stack.size();
        if (count > 0 && oneLess)
        {
            --count;
        }
        for (size_t i = 0; i < count; ++i)
        {
            m_SB.Add(INDENT);
        }
    }
}
#endif // _OA_VMA_JSON_WRITER_FUNCTIONS

namespace
{

void OaVmaPrintDetailedStatistics(OaVmaJsonWriter& json, const OaVmaDetailedStatistics& stat)
{
    json.BeginObject();

    json.WriteString("BlockCount");
    json.WriteNumber(stat.statistics.blockCount);
    json.WriteString("BlockBytes");
    json.WriteNumber(stat.statistics.blockBytes);
    json.WriteString("AllocationCount");
    json.WriteNumber(stat.statistics.allocationCount);
    json.WriteString("AllocationBytes");
    json.WriteNumber(stat.statistics.allocationBytes);
    json.WriteString("UnusedRangeCount");
    json.WriteNumber(stat.unusedRangeCount);

    if (stat.statistics.allocationCount > 1)
    {
        json.WriteString("AllocationSizeMin");
        json.WriteNumber(stat.allocationSizeMin);
        json.WriteString("AllocationSizeMax");
        json.WriteNumber(stat.allocationSizeMax);
    }
    if (stat.unusedRangeCount > 1)
    {
        json.WriteString("UnusedRangeSizeMin");
        json.WriteNumber(stat.unusedRangeSizeMin);
        json.WriteString("UnusedRangeSizeMax");
        json.WriteNumber(stat.unusedRangeSizeMax);
    }
    json.EndObject();
}

} // namespace

#endif // _OA_VMA_JSON_WRITER
